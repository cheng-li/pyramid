package edu.neu.ccs.pyramid.regression.linear_regression;

import edu.neu.ccs.pyramid.dataset.DataSet;
import edu.neu.ccs.pyramid.dataset.RegDataSet;
import edu.neu.ccs.pyramid.optimization.Terminator;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.apache.mahout.math.Vector;

import java.util.Arrays;
import java.util.stream.IntStream;

/**
 * Friedman, Jerome, Trevor Hastie, and Rob Tibshirani.
 * "Regularization paths for generalized linear models via coordinate descent."
 * Journal of statistical software 33.1 (2010): 1.
 * Created by chengli on 2/18/15.
 */
public class ElasticNetLinearRegOptimizer {
    private static final Logger logger = LogManager.getLogger();
    private double regularization = 0;
    private double l1Ratio = 0;
    private Terminator terminator;
    private LinearRegression linearRegression;
    private DataSet dataSet;
    private double[] labels;
    double[] instanceWeights;

    public ElasticNetLinearRegOptimizer(LinearRegression linearRegression, DataSet dataSet, double[] labels, double[] instanceWeights) {
        this.linearRegression = linearRegression;
        this.dataSet = dataSet;
        this.labels = labels;
        this.instanceWeights = instanceWeights;
        this.terminator = new Terminator();
    }

    public ElasticNetLinearRegOptimizer(LinearRegression linearRegression, DataSet dataSet, double[] labels) {
        this(linearRegression,dataSet,labels,defaultWeights(dataSet.getNumDataPoints()));
    }

    public ElasticNetLinearRegOptimizer(LinearRegression linearRegression, RegDataSet dataSet) {
        this(linearRegression,dataSet,dataSet.getLabels());
    }

    public double getRegularization() {
        return regularization;
    }

    public void setRegularization(double regularization) {
        this.regularization = regularization;
    }

    public double getL1Ratio() {
        return l1Ratio;
    }

    public void setL1Ratio(double l1Ratio) {
        this.l1Ratio = l1Ratio;
    }

    public Terminator getTerminator() {
        return terminator;
    }



    /**
     * weighted least square fit by coordinate descent
     */
    public void optimize(){
        double[] scores = new double[dataSet.getNumDataPoints()];
        IntStream.range(0,dataSet.getNumDataPoints()).parallel().forEach(i->
            scores[i] = linearRegression.predict(dataSet.getRow(i)));

        double lastLoss = loss(linearRegression,scores,labels,instanceWeights);
        if (logger.isDebugEnabled()){
            logger.debug("initial loss = "+lastLoss);
        }

        while(true){
            iterate(scores);
            double loss = loss(linearRegression,scores,labels,instanceWeights);
            if (logger.isDebugEnabled()){
                logger.debug("loss = "+loss);
            }
            terminator.add(loss);
            if (terminator.shouldTerminate()){
                if (logger.isDebugEnabled()){
                    logger.debug("final loss = "+loss);
                }
                break;
            }
        }
    }

    /**
     * one cycle of coordinate descent
     */
    private void iterate(double[] scores){
        double totalWeight = Arrays.stream(instanceWeights).parallel().sum();
        // if no weight at all, only minimize the penalty
        if (totalWeight==0){
            // if there is a penalty
            if (regularization>0){
                for (int j=0;j<dataSet.getNumFeatures();j++){
                    linearRegression.getWeights().setWeight(j,0);
                }
            }
            return;
        }
        double oldBias = linearRegression.getWeights().getBias();
        double newBias = IntStream.range(0,dataSet.getNumDataPoints()).parallel().mapToDouble(i ->
        instanceWeights[i]*(labels[i]-scores[i] + oldBias)).sum()/totalWeight;
        linearRegression.getWeights().setBias(newBias);
        //update scores
        double difference = newBias - oldBias;
        IntStream.range(0,dataSet.getNumDataPoints()).parallel().forEach(i -> scores[i] = scores[i] + difference);
        for (int j=0;j<dataSet.getNumFeatures();j++){
            optimizeOneFeature(scores,j);
        }
    }

    private void optimizeOneFeature(double[] scores, int featureIndex){
        double oldCoeff = linearRegression.getWeights().getWeightsWithoutBias().get(featureIndex);
        double fit = 0;
        double denominator = 0;
        Vector featureColumn = dataSet.getColumn(featureIndex);
        for (Vector.Element element: featureColumn.nonZeroes()){
            int i = element.index();
            double x = element.get();
            double partialResidual = labels[i] - scores[i] + x*oldCoeff;
            fit += instanceWeights[i]*x*partialResidual;
            denominator += x*x*instanceWeights[i];
        }
        double numerator = softThreshold(fit);
        denominator += regularization*(1-l1Ratio);
        // if denominator = 0, this feature is useless, assign 0 to the coefficient
        double newCoeff = 0;
        if (denominator!=0){
            newCoeff = numerator/denominator;
        }


        linearRegression.getWeights().setWeight(featureIndex,newCoeff);
        //update scores
        double difference = newCoeff - oldCoeff;
        if (difference!=0){
            for (Vector.Element element: featureColumn.nonZeroes()){
                int i = element.index();
                double x = element.get();
                scores[i] = scores[i] +  difference*x;
            }
        }
    }


    private double loss(LinearRegression linearRegression, double[] scores, double[] labels, double[] instanceWeights){
        double mse = IntStream.range(0,scores.length).parallel().mapToDouble(i ->
                0.5 * instanceWeights[i] * Math.pow(labels[i] - scores[i], 2))
                .sum();
        double penalty = penalty(linearRegression);
        return mse + penalty;
    }


    private double penalty(LinearRegression linearRegression){
        Vector vector = linearRegression.getWeights().getWeightsWithoutBias();
        double normCombination = (1-l1Ratio)*0.5*Math.pow(vector.norm(2),2) +
                l1Ratio*vector.norm(1);
        return regularization * normCombination;
    }

    private static double softThreshold(double z, double gamma){
        if (z>0 && gamma < Math.abs(z)){
            return z-gamma;
        }
        if (z<0 && gamma < Math.abs(z)){
            return z+gamma;
        }
        return 0;
    }

    private double softThreshold(double z){
        return softThreshold(z, regularization*l1Ratio);
    }

    // todo double check: what's the meaning of weight; what happens if default weight = 1; how will that affect hyper parameters?
    private static double[] defaultWeights(int numData){
        double[] weights = new double[numData];
        double weight = 1.0/numData;
        Arrays.fill(weights,weight);
        return weights;
    }



}
