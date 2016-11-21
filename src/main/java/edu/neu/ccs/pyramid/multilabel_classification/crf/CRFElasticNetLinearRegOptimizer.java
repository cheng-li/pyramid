package edu.neu.ccs.pyramid.multilabel_classification.crf;

import edu.neu.ccs.pyramid.dataset.DataSet;
import edu.neu.ccs.pyramid.optimization.Terminator;
import org.apache.mahout.math.Vector;

import java.util.Arrays;
import java.util.stream.IntStream;

/**
 * Created by Rainicy on 11/12/16.
 */
public class CRFElasticNetLinearRegOptimizer {
    private double regularization = 0;
    private double l1Ratio = 0;
    private Terminator terminator;
    private DataSet dataSet;
    private double[] labels;
    double[] instanceWeights;
    double sumWeights;
    private CRFLinearRegression linearRegression;

    public CRFElasticNetLinearRegOptimizer(CRFLinearRegression linearRegression, DataSet dataSet, double[] labels, double[] instanceWeights, double sumWeights) {
        this.linearRegression = linearRegression;
        this.dataSet = dataSet;
        this.labels = labels;
        this.instanceWeights = instanceWeights;
        this.terminator = new Terminator();
        this.sumWeights = sumWeights;
    }

    public CRFElasticNetLinearRegOptimizer(CRFLinearRegression linearRegression, DataSet dataSet, double[] labels, double[] instanceWeights) {
        this(linearRegression, dataSet, labels, instanceWeights, Arrays.stream(instanceWeights).parallel().sum());
    }

    public CRFElasticNetLinearRegOptimizer(CRFLinearRegression linearRegression, DataSet dataSet, double[] labels) {
        this(linearRegression,dataSet,labels,defaultWeights(dataSet.getNumDataPoints()));
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

    public void optimize(){
        double[] scores = new double[dataSet.getNumDataPoints()];
        IntStream.range(0,dataSet.getNumDataPoints()).parallel().forEach(i->
                scores[i] = linearRegression.predict(dataSet.getRow(i)));

//        System.out.println("one: ");
        while(true){
            iterate(scores);
            double loss = loss(linearRegression,scores,labels,instanceWeights,sumWeights);
//            System.out.println("loss: " + loss);
            terminator.add(loss);
            if (terminator.shouldTerminate()){
                break;
            }
        }
    }

    private void iterate(double[] scores){
        // if no weight at all, only minimize the penalty
        if (sumWeights==0){
            // if there is a penalty
            if (regularization>0){
                for (int j=0;j<dataSet.getNumFeatures();j++){
                    linearRegression.getWeights().setWeight(j,0);
                }
            }
            return;
        }
        for (int j=0;j<dataSet.getNumFeatures();j++){
            optimizeOneFeature(scores,j);
        }
    }

    private void optimizeOneFeature(double[] scores, int featureIndex){
        double oldCoeff = linearRegression.getWeights().getWeights().get(featureIndex);
        double fit = 0;
        double denominator = 0;
        Vector featureColumn = dataSet.getColumn(featureIndex);
        for (Vector.Element element: featureColumn.nonZeroes()){
            int i = element.index();
            double x = element.get();
            double partialResidual = labels[i] - scores[i] + x*oldCoeff;
            double tmp = instanceWeights[i]*x;
            fit += tmp*partialResidual;
            denominator += x*tmp;
        }
        fit /= sumWeights;
        double numerator = softThreshold(fit);
        denominator = denominator/sumWeights + regularization*(1-l1Ratio);
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


    private double loss(CRFLinearRegression linearRegression, double[] scores, double[] labels, double[] instanceWeights, double sumWeights){
        double mse = IntStream.range(0,scores.length).parallel().mapToDouble(i ->
                instanceWeights[i] * Math.pow(labels[i] - scores[i], 2))
                .sum();
        double penalty = penalty(linearRegression);
        return mse/(2*sumWeights) + penalty;
    }

    private double penalty(CRFLinearRegression linearRegression){
        Vector vector = linearRegression.getWeights().getWeights();
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

    private static double[] defaultWeights(int numData){
        double[] weights = new double[numData];
        double weight = 1.0;
        Arrays.fill(weights,weight);
        return weights;
    }
}
