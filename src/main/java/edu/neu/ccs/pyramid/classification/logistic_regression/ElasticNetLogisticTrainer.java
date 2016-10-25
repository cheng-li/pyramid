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

import java.util.Arrays;
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
    // instances weights
    private double[] weights;
    private double sumWeights;
    private double regularization;
    private double l1Ratio;
    // relative threshold
    private double epsilon;
    private Vector empiricalCounts;
    private Vector predictedCounts;
    private int numParameters;
    // size num classes * num data
    private double[][] probabilityMatrix;
    private Terminator terminator;
    private boolean lineSearch = true;

    public static Builder newBuilder(LogisticRegression logisticRegression, DataSet dataSet, int numClasses,
                                     double[][] targets, double[] weights) {
        return new Builder(logisticRegression, dataSet, numClasses, targets, weights);
    }

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

    public void iterate() {
        double[][] probs = new double[dataSet.getNumDataPoints()][numClasses];
        double[][] classScores = new double[dataSet.getNumDataPoints()][numClasses];
        IntStream.range(0, dataSet.getNumDataPoints()).parallel().forEach(i -> {
            probs[i] = logisticRegression.predictClassProbs(dataSet.getRow(i));
            classScores[i] = logisticRegression.predictClassScores(dataSet.getRow(i));
        });
        Weights oldWeights = logisticRegression.getWeights().deepCopy();
        IntStream.range(0,numClasses).parallel().forEach(i -> optimizeOneClass(i, probs, classScores));
        if (lineSearch) {
            Weights newWeights = logisticRegression.getWeights().deepCopy();
            // infer searchDirection
            Vector searchDirection = newWeights.getAllWeights().minus(oldWeights.getAllWeights());

            if (logger.isDebugEnabled()){
                logger.debug("norm of the search direction = " + searchDirection.norm(2));
            }
            // move back to starting point
            logisticRegression.getWeights().setWeightVector(oldWeights.getAllWeights());
            // this gradient doesn't include the penalty term, so it is only approximate
            Vector gradient = this.predictedCounts.minus(empiricalCounts).divide(dataSet.getNumDataPoints());
            lineSearch(searchDirection, gradient);
            updateClassProbMatrix();
            updatePredictedCounts();
        }
        terminator.add(getLoss());
    }

    private void optimizeOneClass(int classIndex, double[][] probs, double[][] classScores) {
        //create weighted least square problem
        int numDataPoints = dataSet.getNumDataPoints();
        double[] realLabels = new double[numDataPoints];
        double[] instanceWeights = new double[numDataPoints];
        IntStream.range(0,numDataPoints).parallel().forEach(i ->
        {
            // TODO: repeated calculations in following two steps.
            double prob = probs[i][classIndex];
            double classScore = classScores[i][classIndex];
            double y = targets[i][classIndex];

            double frac = 0;
            double tmpP = prob*(1-prob);
            // if prob = 0 or prob = 1, weight = 0; doesn't matter how we decide frac; leave it 0
            if (prob!=0&&prob!=1){
                frac = (y-prob)/tmpP;
            }
            // frac is numerically unstable; if it is too big, the weighted least square solver will crash
            if (frac>1){
                frac=1;
            }

            if (frac<-1){
                frac=-1;
            }

            realLabels[i] = classScore + frac;
            instanceWeights[i] = weights[i]*tmpP;
//            instanceWeights[i] = weights[i]*tmpP;
        });

        // in glmnet algorithm:
        // this correspond to moving towards the search direction with step size 1
        LinearRegression linearRegression = new LinearRegression(dataSet.getNumFeatures(),
                logisticRegression.getWeights().getWeightsForClass(classIndex));
        // use default epsilon
        ElasticNetLinearRegOptimizer linearRegTrainer = new ElasticNetLinearRegOptimizer(linearRegression,dataSet,realLabels,instanceWeights, sumWeights);
        linearRegTrainer.setRegularization(this.regularization);
        linearRegTrainer.setL1Ratio(this.l1Ratio);
        if (logger.isDebugEnabled()){
            logger.debug("start linearRegTrainer.optimize()");
        }
        linearRegTrainer.optimize();
        if (logger.isDebugEnabled()){
            logger.debug("finish linearRegTrainer.optimize()");
        }

        if (logger.isDebugEnabled()){
            logger.debug("loss after optimization of one class = " + loss());
        }
    }


    public void iterate1(){
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
            double tmpP = prob*(1-prob);
            // if prob = 0 or prob = 1, weight = 0; doesn't matter how we decide frac; leave it 0
            if (prob!=0&&prob!=1){
                frac = (y-prob)/tmpP;
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
//            instanceWeights[i] = (weights[i]*prob*(1-prob))/numDataPoints;
            instanceWeights[i] = (weights[i]*tmpP);
        });

        Weights oldWeights = logisticRegression.getWeights().deepCopy();

        // in glmnet algorithm:
        // this correspond to moving towards the search direction with step size 1
        // TODO: use the oldWeights
        LinearRegression linearRegression = new LinearRegression(dataSet.getNumFeatures(),
                logisticRegression.getWeights().getWeightsForClass(classIndex));
        // use default epsilon
        ElasticNetLinearRegOptimizer linearRegTrainer = new ElasticNetLinearRegOptimizer(linearRegression,dataSet,realLabels,instanceWeights,sumWeights);
        linearRegTrainer.setRegularization(this.regularization);
        linearRegTrainer.setL1Ratio(this.l1Ratio);
        if (logger.isDebugEnabled()){
            logger.debug("start linearRegTrainer.optimize()");
        }
        linearRegTrainer.optimize();
        if (logger.isDebugEnabled()){
            logger.debug("finish linearRegTrainer.optimize()");
        }

        if (lineSearch) {
            Weights newWeights = logisticRegression.getWeights().deepCopy();
            // infer searchDirection
            Vector searchDirection = newWeights.getAllWeights().minus(oldWeights.getAllWeights());

            if (logger.isDebugEnabled()){
                logger.debug("norm of the search direction = " + searchDirection.norm(2));
            }
            // move back to starting point
            logisticRegression.getWeights().setWeightVector(oldWeights.getAllWeights());
            // this gradient doesn't include the penalty term, so it is only approximate
            Vector gradient = this.predictedCounts.minus(empiricalCounts).divide(numDataPoints);
            lineSearch(searchDirection, gradient);
            updatePredictedCounts();
            updateClassProbMatrix();
        }

        if (logger.isDebugEnabled()){
            logger.debug("loss after optimization of one class = " + loss());
        }
    }


    private void updateClassProbs(int dataPointIndex){
        double[] probs = logisticRegression.predictClassProbs(dataSet.getRow(dataPointIndex));
        for (int k=0;k<numClasses;k++){
            this.probabilityMatrix[k][dataPointIndex]=probs[k];
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
        // todo: this should be re-implemented here
        // should not use the method provided by LR
        // negativeLogLikelihood should be multiplied by weights
        double negativeLogLikelihood = logisticRegression.dataSetLogLikelihood(dataSet, targets, weights) * -1;
        double penalty = penalty();
//        return negativeLogLikelihood/dataSet.getNumDataPoints() + penalty;
        return negativeLogLikelihood/sumWeights + penalty;
    }

    private double loss(double penalty){
        double negativeLogLikelihood = logisticRegression.dataSetLogLikelihood(dataSet, targets, weights) * -1;
//        return negativeLogLikelihood/dataSet.getNumDataPoints() + penalty;
        return negativeLogLikelihood/sumWeights + penalty;
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


//    private void withoutLineSearch(Vector searchDirection) {
//        Vector start = logisticRegression.getWeights().getAllWeights();
//        Vector target = start.plus(searchDirection);
//        logisticRegression.getWeights().setWeightVector(target);
//    }

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

        localSearchDir = searchDirection;

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
        double[] probs = this.probabilityMatrix[classIndex];
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
        // N
        private double[] weights;
        private double sumWeights;
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
            this.weights = new double[dataSet.getNumDataPoints()];
            Arrays.fill(this.weights, 1);
            this.sumWeights = Arrays.stream(weights).parallel().sum();
        }

        public Builder(LogisticRegression logisticRegression, ClfDataSet dataSet) {
            this(logisticRegression, dataSet, dataSet.getNumClasses(), dataSet.getLabels());
        }

        public Builder(LogisticRegression logisticRegression, DataSet dataSet, int numClasses, double[][] targets) {
            this.logisticRegression = logisticRegression;
            this.dataSet = dataSet;
            this.numClasses = numClasses;
            this.targets = targets;
            this.weights = new double[dataSet.getNumDataPoints()];
            Arrays.fill(this.weights, 1);
            this.sumWeights = Arrays.stream(weights).parallel().sum();
        }

        public Builder(LogisticRegression logisticRegression, DataSet dataSet, int numClasses, double[][] targets,
                       double[] weights) {
            this.logisticRegression = logisticRegression;
            this.dataSet = dataSet;
            this.numClasses = numClasses;
            this.targets = targets;
            this.weights = weights;
            this.sumWeights = Arrays.stream(weights).parallel().sum();
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
            trainer.weights = weights;
            trainer.sumWeights = sumWeights;
            trainer.numClasses = numClasses;
            trainer.regularization = this.regularization;
            trainer.l1Ratio = this.l1Ratio;
            trainer.epsilon = this.epsilon;
            trainer.lineSearch = this.lineSearch;
            trainer.numParameters = logisticRegression.getWeights().totalSize();
            trainer.empiricalCounts = new DenseVector(trainer.numParameters);
            trainer.predictedCounts = new DenseVector(trainer.numParameters);
            trainer.probabilityMatrix = new double[numClasses][dataSet.getNumDataPoints()];
            trainer.updateEmpricalCounts();
            trainer.updateClassProbMatrix();
            trainer.updatePredictedCounts();
            trainer.terminator = new Terminator();
            return trainer;
        }
    }
}
