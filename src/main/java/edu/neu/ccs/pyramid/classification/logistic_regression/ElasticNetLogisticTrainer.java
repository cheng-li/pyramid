package edu.neu.ccs.pyramid.classification.logistic_regression;

import edu.neu.ccs.pyramid.dataset.ClfDataSet;
import edu.neu.ccs.pyramid.dataset.DataSet;
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
    private DataSet dataSet;
    private int numClasses;
//    private int[] labels;
    // y_nl: number of datapoint and number of labels
    private double[][] targets;
    private double regularization;
    private double l1Ratio;
    // relative threshold
    private double epsilon;
    private Vector empiricalCounts;
    private Vector predictedCounts;
    private int numParameters;
    private ProbabilityMatrix probabilityMatrix;
    private Terminator terminator;
    private boolean lineSearch;

    public static Builder newBuilder(LogisticRegression logisticRegression, DataSet dataSet, int numClasses, double[][] targets) {
        return new Builder(logisticRegression, dataSet, numClasses, targets);
    }

    public static Builder newBuilder(LogisticRegression logisticRegression, DataSet dataSet, int numClasses, int[] labels){
        return new Builder(logisticRegression, dataSet, numClasses, labels);
    }

    public static Builder newBuilder(LogisticRegression logisticRegression, ClfDataSet dataSet){
        return new Builder(logisticRegression, dataSet);
    }

    public void optimize(){
        logisticRegression.setFeatureList(dataSet.getFeatureList());
//        logisticRegression.setLabelTranslator(dataSet.getLabelTranslator());

        while(true){
            iterate();
            if (terminator.shouldTerminate()){
                break;
            }
        }
    }

    public void iterate(){
        for (int k=0;k<numClasses;k++){
            optimizeOneClass(k);
        }
        terminator.add(getLoss());
    }

    public double getLoss(){
        return loss();
    }

    public Terminator getTerminator() {
        return terminator;
    }

    private void optimizeOneClass(int classIndex){
        //create weighted least square problem
        int numDataPoints = dataSet.getNumDataPoints();
        double[] realLabels = new double[numDataPoints];
        double[] instanceWeights = new double[numDataPoints];
        IntStream.range(0,numDataPoints).parallel().forEach(i ->
        {
            // TODO: repeated calculations in following two steps.
            double prob = logisticRegression.predictClassProbs(dataSet.getRow(i))[classIndex];
            double classScore = logisticRegression.predictClassScore(dataSet.getRow(i),classIndex);
//            double y = 0;
            double y = targets[i][classIndex];
//            if (labels[i]==classIndex){
//                y = 1;
//            }
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

            realLabels[i] = classScore + frac;
            // TODO: why divided by numDataPoints?
            instanceWeights[i] = (prob*(1-prob))/numDataPoints;
        });

        Weights oldWeights = logisticRegression.getWeights().deepCopy();

        // this gradient doesn't include the penalty term, so it is only approximate
        Vector gradient = this.predictedCounts.minus(empiricalCounts).divide(numDataPoints);

        // in glmnet algorithm:
        // this correspond to moving towards the search direction with step size 1
        // TODO: use the oldWeights
        LinearRegression linearRegression = new LinearRegression(dataSet.getNumFeatures(),
                logisticRegression.getWeights().getWeightsForClass(classIndex));
        // use default epsilon
        ElasticNetLinearRegOptimizer linearRegTrainer = new ElasticNetLinearRegOptimizer(linearRegression,dataSet,realLabels,instanceWeights);
        linearRegTrainer.setRegularization(this.regularization);
        linearRegTrainer.setL1Ratio(this.l1Ratio);
        if (logger.isDebugEnabled()){
            logger.debug("start linearRegTrainer.optimize()");
        }
        linearRegTrainer.optimize();
        if (logger.isDebugEnabled()){
            logger.debug("finish linearRegTrainer.optimize()");
        }

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
        if (lineSearch) {
            lineSearch(searchDirection, gradient);
        } else {
            withoutLineSearch(searchDirection, gradient);
        }

        if (logger.isDebugEnabled()){
            logger.debug("loss after optimization of one class = " + loss());
        }

        updateClassProbMatrix();
        updatePredictedCounts();
    }


    private void updateClassProbs(int dataPointIndex){
        double[] probs = logisticRegression.predictClassProbs(dataSet.getRow(dataPointIndex));
        for (int k=0;k<numClasses;k++){
            this.probabilityMatrix.setProbability(dataPointIndex,k,probs[k]);
        }
    }

    private void updateClassProbMatrix(){
        if (logger.isDebugEnabled()){
            logger.debug("start updateClassProbMatrix()");
        }
        IntStream.range(0,dataSet.getNumDataPoints()).parallel()
                .forEach(this::updateClassProbs);
        if (logger.isDebugEnabled()){
            logger.debug("finish updateClassProbMatrix()");
        }
    }

    private double loss(){
        double negativeLogLikelihood = logisticRegression.dataSetLogLikelihood(dataSet, targets) * -1;
        double penalty = penalty();
        return negativeLogLikelihood/dataSet.getNumDataPoints() + penalty;
    }

    private double loss(double penalty){
        double negativeLogLikelihood = logisticRegression.dataSetLogLikelihood(dataSet, targets) * -1;
        return negativeLogLikelihood/dataSet.getNumDataPoints() + penalty;
    }


    private double penalty(){
        return IntStream.range(0,logisticRegression.getNumClasses()).parallel().mapToDouble(k -> penalty(k)).sum();
    }

    private double penalty(int k) {
        Vector vector = logisticRegression.getWeights().getWeightsWithoutBiasForClass(k);
        double normCombination = (1-l1Ratio)*0.5*Math.pow(vector.norm(2),2) +
                l1Ratio*vector.norm(1);
        return regularization * normCombination;
    }


    private void withoutLineSearch(Vector searchDirection, Vector gradient) {
        Vector localSearchDir;
        double product = gradient.dot(searchDirection);
        if (product < 0){
            localSearchDir = searchDirection;
        } else {
            if (logger.isWarnEnabled()) {
                logger.warn("Bad search direction! Use negative gradient instead. Product of gradient and search direction = " + product);
            }
            localSearchDir = gradient.times(-1);
        }

        Vector start = logisticRegression.getWeights().getAllWeights();

        Vector target = start.plus(localSearchDir);
        logisticRegression.getWeights().setWeightVector(target);
    }

    /**
     * a special back track line search for sufficient decrease with elasticnet penalized model
     * reference:
     * An improved glmnet for l1-regularized logistic regression.
     * @param searchDirection
     * @return
     */
    private void lineSearch(Vector searchDirection, Vector gradient){
        Vector localSearchDir;
        double initialStepLength = 1;
        double shrinkage = 0.5;
        double c = 1e-4;
        double stepLength = initialStepLength;
        Vector start = logisticRegression.getWeights().getAllWeights();
        double penalty = penalty();
        double value = loss(penalty);
        if (logger.isDebugEnabled()){
            logger.debug("start line search");
            logger.debug("initial loss = "+loss());
        }
        double product = gradient.dot(searchDirection);
        if (product < 0){
            localSearchDir = searchDirection;
        } else {
            if (logger.isWarnEnabled()) {
                logger.warn("Bad search direction! Use negative gradient instead. Product of gradient and search direction = " + product);
            }

            localSearchDir = gradient.times(-1);
        }

        while(true){
            Vector step = localSearchDir.times(stepLength);
            Vector target = start.plus(step);
            logisticRegression.getWeights().setWeightVector(target);
            double targetPenalty = penalty();
            double targetValue = loss(targetPenalty);
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
        int featureIndex = logisticRegression.getWeights().getFeatureIndex(parameterIndex);
        double count = 0;
        //bias
        if (featureIndex == -1){
            for (int i=0;i<dataSet.getNumDataPoints();i++){
                count += targets[i][classIndex];
//                if (labels[i]==classIndex){
//                    count +=1;
//                }
            }
        } else {
            Vector featureColumn = dataSet.getColumn(featureIndex);
            for (Vector.Element element: featureColumn.nonZeroes()){
                int dataPointIndex = element.index();
                double featureValue = element.get();
                count += featureValue * targets[dataPointIndex][classIndex];
//                int label = labels[dataPointIndex];
//                if (label==classIndex){
//                    count += featureValue;
//                }
            }
        }
        return count;
    }

    private void updatePredictedCounts(){
        if (logger.isDebugEnabled()){
            logger.debug("start updatePredictedCounts()");
        }
        IntStream.range(0,numParameters).parallel()
                .forEach(i -> this.predictedCounts.set(i, calPredictedCount(i)));
        if (logger.isDebugEnabled()){
            logger.debug("finish updatePredictedCounts()");
        }
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
        private DataSet dataSet;
//        private int[] labels;
        // N * L
        private double[][] targets;
        private int numClasses;

        // when p>>N, logistic regression with 0 regularization is ill-defined
        // use a small regularization
        private double regularization=0.00001;
        private double l1Ratio=0;
        private double epsilon=0.001;
        private boolean lineSearch=true;


        public Builder(LogisticRegression logisticRegression, DataSet dataSet, int numClasses, int[] labels) {
            int numDataPoints = dataSet.getNumDataPoints();
            double[][] targs = new double[numDataPoints][numClasses];
            for (int i=0; i<numDataPoints; i++) {
                targs[i][labels[i]] = 1.0;
            }

            this.logisticRegression = logisticRegression;
            this.dataSet = dataSet;
            this.numClasses = numClasses;
            this.targets = targs;
        }

        public Builder(LogisticRegression logisticRegression, ClfDataSet dataSet) {
            this(logisticRegression, dataSet, dataSet.getNumClasses(), dataSet.getLabels());
        }

        public Builder(LogisticRegression logisticRegression, DataSet dataSet, int numClasses, double[][] targets) {
            this.logisticRegression = logisticRegression;
            this.dataSet = dataSet;
            this.numClasses = numClasses;
            this.targets = targets;
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

        public Builder setLineSearch(boolean lineSearch) {
            this.lineSearch = lineSearch;
            return this;
        }

        public ElasticNetLogisticTrainer build(){
            ElasticNetLogisticTrainer trainer = new ElasticNetLogisticTrainer();
            trainer.logisticRegression = logisticRegression;
            trainer.dataSet = dataSet;
            trainer.targets = targets;
            trainer.numClasses = numClasses;
            trainer.regularization = this.regularization;
            trainer.l1Ratio = this.l1Ratio;
            trainer.epsilon = this.epsilon;
            trainer.lineSearch = this.lineSearch;
            trainer.numParameters = logisticRegression.getWeights().totalSize();
            trainer.empiricalCounts = new DenseVector(trainer.numParameters);
            trainer.predictedCounts = new DenseVector(trainer.numParameters);
            trainer.probabilityMatrix = new ProbabilityMatrix(dataSet.getNumDataPoints(),numClasses);
            trainer.updateEmpricalCounts();
            trainer.updateClassProbMatrix();
            trainer.updatePredictedCounts();
            trainer.terminator = new Terminator();
            return trainer;
        }
    }
}
