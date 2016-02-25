package edu.neu.ccs.pyramid.classification.logistic_regression;

import edu.neu.ccs.pyramid.dataset.ClfDataSet;
import edu.neu.ccs.pyramid.dataset.ProbabilityMatrix;
import edu.neu.ccs.pyramid.optimization.*;
import edu.neu.ccs.pyramid.regression.linear_regression.*;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.Vector;

import java.util.stream.IntStream;

/**
 * Friedman, Jerome, Trevor Hastie, and Rob Tibshirani.
 * "Regularization paths for generalized linear models via coordinate descent."
 * Journal of statistical software 33.1 (2010): 1.
 *
 * Yuan, Guo-Xun, Chia-Hua Ho, and Chih-Jen Lin.
 * "An improved glmnet for l1-regularized logistic regression."
 * The Journal of Machine Learning Research 13.1 (2012): 1999-2030.
 *
 * Dan Klein and Chris Manning.
 * "Maxent Models, Conditional Estimation, and Optimization, without the Magic."
 *
 * Created by chengli on 2/24/15.
 */
public class ElasticNetLogisticTrainer {
    private static final Logger logger = LogManager.getLogger();
    private LogisticRegression logisticRegression;
    private ClfDataSet dataSet;
    private double regularization;
    private double l1Ratio;
    // relative threshold
    private double epsilon;
    private Vector empiricalCounts;
    private Vector predictedCounts;
    private int numParameters;
    private ProbabilityMatrix probabilityMatrix;
    private Terminator terminator;

    public static Builder newBuilder(LogisticRegression logisticRegression, ClfDataSet dataSet){
        return new Builder(logisticRegression, dataSet);
    }

    public void optimize(){
        logisticRegression.setFeatureList(dataSet.getFeatureList());
        logisticRegression.setLabelTranslator(dataSet.getLabelTranslator());

        while(true){
            iterate();
            if (terminator.shouldTerminate()){
                break;
            }
        }
    }

    public void iterate(){
        for (int k=0;k<dataSet.getNumClasses();k++){
            optimizeOneClass(k);
        }
        terminator.add(getLoss());
    }

    public double getLoss(){
        return loss();
    }



    private void optimizeOneClass(int classIndex){
        //create weighted least square problem
        int numDataPoints = dataSet.getNumDataPoints();
        double[] labels = new double[numDataPoints];
        double[] instanceWeights = new double[numDataPoints];
        IntStream.range(0,numDataPoints).parallel().forEach(i ->
        {
            double prob = logisticRegression.predictClassProbs(dataSet.getRow(i))[classIndex];
            double classScore = logisticRegression.predictClassScore(dataSet.getRow(i),classIndex);
            double y = 0;
            if (dataSet.getLabels()[i]==classIndex){
                y = 1;
            }
            double frac = 0;
            // if prob = 0 or prob = 1, weight = 0; doesn't matter how we decide frac; leave it 0
            if (prob!=0&&prob!=1){
                frac = (y-prob)/(prob*(1-prob));
            }
            // frac is numerically unstable; if it is too big, the weighted least square solver will crash
            if (frac>1){
                frac=1;
            }

            if (frac<-1){
                frac=-1;
            }

            labels[i] = classScore + frac;
            instanceWeights[i] = (prob*(1-prob))/numDataPoints;
        });

        Weights oldWeights = logisticRegression.getWeights().deepCopy();

        // this gradient doesn't include the penalty term, so it is only approximate
        Vector gradient = this.predictedCounts.minus(empiricalCounts).divide(numDataPoints);

        // in glmnet algorithm:
        // this correspond to moving towards the search direction with step size 1
        LinearRegression linearRegression = new LinearRegression(dataSet.getNumFeatures(),
                logisticRegression.getWeights().getWeightsForClass(classIndex));
        // use default epsilon
        ElasticNetLinearRegOptimizer linearRegTrainer = new ElasticNetLinearRegOptimizer(linearRegression,dataSet,labels,instanceWeights);
        linearRegTrainer.setRegularization(this.regularization);
        linearRegTrainer.setL1Ratio(this.l1Ratio);
        linearRegTrainer.optimize();

        Weights newWeights = logisticRegression.getWeights().deepCopy();

        // infer searchDirection
        Vector searchDirection = newWeights.getAllWeights().minus(oldWeights.getAllWeights());

        if (logger.isDebugEnabled()){
            logger.debug("norm of the search direction = " + searchDirection.norm(2));
        }
        // move back to starting point
        logisticRegression.getWeights().setWeightVector(oldWeights.getAllWeights());

        // line search
        // the original glmnet algorithm may diverge without line search
        lineSearch(searchDirection, gradient);
        if (logger.isDebugEnabled()){
            logger.debug("loss after optimization of one class = " + loss());
        }

        updateClassProbMatrix();
        updatePredictedCounts();
    }


    private void updateClassProbs(int dataPointIndex){
        double[] probs = logisticRegression.predictClassProbs(dataSet.getRow(dataPointIndex));
        for (int k=0;k<dataSet.getNumClasses();k++){
            this.probabilityMatrix.setProbability(dataPointIndex,k,probs[k]);
        }
    }

    private void updateClassProbMatrix(){
        IntStream.range(0,dataSet.getNumDataPoints()).parallel()
                .forEach(this::updateClassProbs);
    }

    private double loss(){
        double negativeLogLikelihood = logisticRegression.dataSetLogLikelihood(dataSet) * -1;
        double penalty = penalty();
        return negativeLogLikelihood/dataSet.getNumDataPoints() + penalty;
    }


    private double penalty(){
        double penalty = 0;
        for (int k=0;k<logisticRegression.getNumClasses();k++){
            Vector vector = logisticRegression.getWeights().getWeightsWithoutBiasForClass(k);
            double normCombination = (1-l1Ratio)*0.5*Math.pow(vector.norm(2),2) +
                    l1Ratio*vector.norm(1);
            penalty += regularization * normCombination;
        }
        return penalty;
    }

    /**
     * a special back track line search for sufficient decrease with elasticnet penalized model
     * reference:
     * An improved glmnet for l1-regularized logistic regression.
     * @param searchDirection
     * @return
     */
    private void lineSearch(Vector searchDirection, Vector gradient){

        double initialStepLength = 1;
        double shrinkage = 0.5;
        double c = 1e-4;
        double stepLength = initialStepLength;
        Vector start = logisticRegression.getWeights().getAllWeights();
        double value = loss();
        if (logger.isDebugEnabled()){
            logger.debug("start line search");
            logger.debug("initial loss = "+loss());
        }
        double penalty = penalty();
        double product = gradient.dot(searchDirection);
        if (logger.isDebugEnabled()){
            logger.debug("product of search direction and gradient = "+product);
            if (product>0){
                logger.warn("bad search direction for the negative log likelihood term !");
            }
        }

        while(true){
            Vector step = searchDirection.times(stepLength);
            Vector target = start.plus(step);
            logisticRegression.getWeights().setWeightVector(target);
            double targetValue = loss();
            double targetPenalty = penalty();
            if (targetValue <= value + c*stepLength*(product + targetPenalty - penalty)){
                if (logger.isDebugEnabled()){
                    logger.debug("step size = "+stepLength);
                    logger.debug("final loss = "+targetValue);
                    logger.debug("line search done");
                }
                break;
            }
            stepLength *= shrinkage;
        }
    }


    private void updateEmpricalCounts(){
        IntStream.range(0,numParameters).parallel()
                .forEach(i -> this.empiricalCounts.set(i, calEmpricalCount(i)));
    }

    private double calEmpricalCount(int parameterIndex){
        int classIndex = logisticRegression.getWeights().getClassIndex(parameterIndex);
        int[] labels = dataSet.getLabels();
        int featureIndex = logisticRegression.getWeights().getFeatureIndex(parameterIndex);
        double count = 0;
        //bias
        if (featureIndex == -1){
            for (int i=0;i<dataSet.getNumDataPoints();i++){
                if (labels[i]==classIndex){
                    count +=1;
                }
            }
        } else {
            Vector featureColumn = dataSet.getColumn(featureIndex);
            for (Vector.Element element: featureColumn.nonZeroes()){
                int dataPointIndex = element.index();
                double featureValue = element.get();
                int label = labels[dataPointIndex];
                if (label==classIndex){
                    count += featureValue;
                }
            }
        }
        return count;
    }

    private void updatePredictedCounts(){
        IntStream.range(0,numParameters).parallel()
                .forEach(i -> this.predictedCounts.set(i, calPredictedCount(i)));
    }

    private double calPredictedCount(int parameterIndex){
        int classIndex = logisticRegression.getWeights().getClassIndex(parameterIndex);
        int featureIndex = logisticRegression.getWeights().getFeatureIndex(parameterIndex);
        double count = 0;
        double[] probs = this.probabilityMatrix.getProbabilitiesForClass(classIndex);
        //bias
        if (featureIndex == -1){
            for (int i=0;i<dataSet.getNumDataPoints();i++){
                count += probs[i];
            }
        } else {
            Vector featureColumn = dataSet.getColumn(featureIndex);
            for (Vector.Element element: featureColumn.nonZeroes()){
                int dataPointIndex = element.index();
                double featureValue = element.get();
                count += probs[dataPointIndex] * featureValue;
            }
        }
        return count;
    }


    public static class Builder{
        private LogisticRegression logisticRegression;
        private ClfDataSet dataSet;
        // when p>>N, logistic regression with 0 regularization is ill-defined
        // use a small regularization
        private double regularization=0.00001;
        private double l1Ratio=0;
        private double epsilon=0.001;

        public Builder(LogisticRegression logisticRegression, ClfDataSet dataSet) {
            this.logisticRegression = logisticRegression;
            this.dataSet = dataSet;
        }

        public Builder setRegularization(double regularization) {
            boolean legal = regularization>=0;
            if (!legal){
                throw new IllegalArgumentException("regularization>=0");
            }
            this.regularization = regularization;
            return this;
        }


        public Builder setL1Ratio(double l1Ratio) {
            boolean legal = (l1Ratio>=0)&&(l1Ratio<=1);
            if (!legal){
                throw new IllegalArgumentException("(l1Ratio>=0)&&(l1Ratio<=1)");
            }
            this.l1Ratio = l1Ratio;
            return this;
        }

        public Builder setEpsilon(double epsilon) {
            boolean legal = (epsilon>0)&&(epsilon<1);
            if (!legal){
                throw new IllegalArgumentException("(epsilon>0)&&(epsilon<1)");
            }
            this.epsilon = epsilon;
            return this;
        }

        public ElasticNetLogisticTrainer build(){
            ElasticNetLogisticTrainer trainer = new ElasticNetLogisticTrainer();
            trainer.logisticRegression = logisticRegression;
            trainer.dataSet = dataSet;
            trainer.regularization = this.regularization;
            trainer.l1Ratio = this.l1Ratio;
            trainer.epsilon = this.epsilon;
            trainer.numParameters = logisticRegression.getWeights().totalSize();
            trainer.empiricalCounts = new DenseVector(trainer.numParameters);
            trainer.predictedCounts = new DenseVector(trainer.numParameters);
            trainer.probabilityMatrix = new ProbabilityMatrix(dataSet.getNumDataPoints(),dataSet.getNumClasses());
            trainer.updateEmpricalCounts();
            trainer.updateClassProbMatrix();
            trainer.updatePredictedCounts();
            trainer.terminator = new Terminator();
            return trainer;
        }
    }
}
