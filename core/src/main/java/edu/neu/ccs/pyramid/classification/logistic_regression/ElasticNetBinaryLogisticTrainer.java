package edu.neu.ccs.pyramid.classification.logistic_regression;

import edu.neu.ccs.pyramid.dataset.ClfDataSet;
import edu.neu.ccs.pyramid.dataset.DataSet;
import edu.neu.ccs.pyramid.optimization.Terminator;
import edu.neu.ccs.pyramid.regression.linear_regression.ElasticNetLinearRegOptimizer;
import edu.neu.ccs.pyramid.regression.linear_regression.LinearRegression;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.Vector;

import java.util.Arrays;
import java.util.stream.IntStream;

/**
 *
 * special designs for Symmetric Binary Logistic Regression, and
 * the changes are based on ElasticNetLogisticTrainer.
 *
 * Adding the following tricks:
 * 1. Binary Logistic Regression only contains one class weight vector
 *
 * Created by Rainicy on 1/20/20
 */
public class ElasticNetBinaryLogisticTrainer {
    private static final Logger logger = LogManager.getLogger();
    private LogisticRegression logisticRegression;
    private DataSet dataSet;
    // binary case
    private int[] targets; // TODO: assume most of them are 0s
    // instances weights
    private double[] weights;
    private double sumWeights;
    private double regularization;
    private double l1Ratio;
    private Vector empiricalCounts;
    private Vector predictedCounts;
    private int numParameters;
    // size of num data; probability of target=1
    private double[] probabilityMatrix;
    private Terminator terminator;
    private boolean lineSearch = true;

    public boolean isActiveSet() {
        return isActiveSet;
    }

    public void setActiveSet(boolean activeSet) {
        isActiveSet = activeSet;
    }

    private boolean isActiveSet = false;

    private int maxNumLinearRegUpdates = 10;

    public static Builder newBuilder(LogisticRegression logisticRegression, DataSet dataSet, int numClasses,
                                     int[] targets, double[] weights) {
        return new Builder(logisticRegression, dataSet, targets, weights);
    }

    public static Builder newBuilder(LogisticRegression logisticRegression, DataSet dataSet, int[] targets) {
        return new Builder(logisticRegression, dataSet, targets);
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
        optimizeClass();
        terminator.add(getLoss());
    }

    public double getLoss(){
        return loss();
    }

    public Terminator getTerminator() {
        return terminator;
    }

    private void optimizeClass(){
        //create weighted least square problem
        int numDataPoints = dataSet.getNumDataPoints();
        double[] realLabels = new double[numDataPoints];
        double[] instanceWeights = new double[numDataPoints];
        IntStream.range(0,numDataPoints).parallel().forEach(i ->
        {
            // scores[0] = 0; scores[1] =
            double[] classScores = logisticRegression.predictClassScores(dataSet.getRow(i));
            double prob = logisticRegression.predictClassProbs(classScores)[1];
            // TODO: following is calculated twice.
            double y = targets[i];
            double frac = 0;
            double tmpP = prob*(1-prob);
            // if prob = 0 or prob = 1, weight = 0; doesn't matter how we decide frac; leave it 0
            if (tmpP!=0){
                frac = (y-prob)/tmpP;
            }
            // frac is numerically unstable; if it is too big, the weighted least square solver will crash
            if (frac>1){
                frac=1;
            }

            if (frac<-1){
                frac=-1;
            }

            realLabels[i] = classScores[1] + frac;
            instanceWeights[i] = (weights[i]*tmpP);
        });

        Weights oldWeights = null;
        if (lineSearch) {
            oldWeights = logisticRegression.getWeights().deepCopy();
        }

        // in glmnet algorithm:
        // this correspond to moving towards the search direction with step size 1
        // TODO: use the oldWeights
        LinearRegression linearRegression = new LinearRegression(dataSet.getNumFeatures(),
                logisticRegression.getWeights().getAllWeights());
        ElasticNetLinearRegOptimizer linearRegTrainer = new ElasticNetLinearRegOptimizer(linearRegression,dataSet,realLabels,instanceWeights,sumWeights);
        linearRegTrainer.setRegularization(this.regularization);
        linearRegTrainer.setL1Ratio(this.l1Ratio);
        linearRegTrainer.setActiveSet(this.isActiveSet);
        //TODO: no large iterations
        linearRegTrainer.getTerminator().setMaxIteration(maxNumLinearRegUpdates);
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
            // TODO: verify whether divided by sumWeights or numDataPoints
            Vector gradient = this.predictedCounts.minus(empiricalCounts).divide(sumWeights);
            lineSearch(searchDirection, gradient);
            // TODO: updateClassProbMatrix should be first?
            updatePredictedCounts();
            updateClassProbMatrix();
        }

        if (logger.isDebugEnabled()){
            logger.debug("loss after optimization of one class = " + loss());
        }
    }


    /**
     * update the class probs for target=1
     * @param dataPointIndex
     */
    private void updateClassProbs(int dataPointIndex){
        double prob = logisticRegression.predictClassProbs(dataSet.getRow(dataPointIndex))[1];
        this.probabilityMatrix[dataPointIndex]=prob;
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
        return negativeLogLikelihood/sumWeights + penalty;
    }

    private double loss(double penalty){
        double negativeLogLikelihood = logisticRegression.dataSetLogLikelihood(dataSet, targets, weights) * -1;
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

    /**
     * parameterIndex is from 0 to #features;
     * parameterIndex = 0 is for bias;
     * parameterIndex > 0 is the feature index starts from 1
     * @param parameterIndex
     * @return
     */
    private double calEmpricalCount(int parameterIndex){
        double count = 0;
        //bias
        if (parameterIndex == 0){
            for (int i=0;i<dataSet.getNumDataPoints();i++){
                count += targets[i];
            }
        } else {
            Vector featureColumn = dataSet.getColumn(parameterIndex-1);
            for (Vector.Element element: featureColumn.nonZeroes()){
                int dataPointIndex = element.index();
                double featureValue = element.get();
                count += featureValue * targets[dataPointIndex];
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
        double count = 0;
        //bias
        if (parameterIndex == 0){
            for (int i=0;i<dataSet.getNumDataPoints();i++){
                count += this.probabilityMatrix[i];
            }
        } else {
            Vector featureColumn = dataSet.getColumn(parameterIndex-1);
            for (Vector.Element element: featureColumn.nonZeroes()){
                int dataPointIndex = element.index();
                double featureValue = element.get();
                count += this.probabilityMatrix[dataPointIndex] * featureValue;
            }
        }
        return count;
    }


    public static class Builder{
        private LogisticRegression logisticRegression;
        private DataSet dataSet;
        // N
        private int[] targets;
        // N
        private double[] weights;
        private double sumWeights;

        // when p>>N, logistic regression with 0 regularization is ill-defined
        // use a small regularization
        private double regularization=0.00001;
        private double l1Ratio=0;
        private boolean lineSearch=true;

        private int maxNumLinearRegUpdates=10;


        public Builder(LogisticRegression logisticRegression, ClfDataSet dataSet) {
            this(logisticRegression, dataSet, dataSet.getLabels());
        }

        /**
         * without weights for isWeighted = False
         * @param logisticRegression
         * @param dataSet
         * @param targets
         */
        public Builder(LogisticRegression logisticRegression, DataSet dataSet, int[] targets) {
            this.logisticRegression = logisticRegression;
            this.dataSet = dataSet;
            this.targets = targets;
            this.weights = new double[dataSet.getNumDataPoints()];
            Arrays.fill(this.weights, 1);
            this.sumWeights = Arrays.stream(weights).parallel().sum();
        }

        public Builder(LogisticRegression logisticRegression, DataSet dataSet, int[] targets,
                       double[] weights) {
            this.logisticRegression = logisticRegression;
            this.dataSet = dataSet;
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


        public Builder setLineSearch(boolean lineSearch) {
            this.lineSearch = lineSearch;
            return this;
        }

        public Builder setMaxNumLinearRegUpdates(int maxNumLinearRegUpdates) {
            this.maxNumLinearRegUpdates = maxNumLinearRegUpdates;
            return this;
        }

        public ElasticNetBinaryLogisticTrainer build(){
            ElasticNetBinaryLogisticTrainer trainer = new ElasticNetBinaryLogisticTrainer();
            trainer.logisticRegression = logisticRegression;
            trainer.dataSet = dataSet;
            trainer.targets = targets;
            trainer.weights = weights;
            trainer.sumWeights = sumWeights;
            trainer.regularization = this.regularization;
            trainer.l1Ratio = this.l1Ratio;
            trainer.lineSearch = this.lineSearch;
            trainer.numParameters = logisticRegression.getWeights().totalSize();

            if (this.lineSearch) {
                trainer.empiricalCounts = new DenseVector(trainer.numParameters);
                trainer.predictedCounts = new DenseVector(trainer.numParameters);
                trainer.probabilityMatrix = new double[dataSet.getNumDataPoints()];
                trainer.updateEmpricalCounts(); // updated only once
                trainer.updateClassProbMatrix();
                trainer.updatePredictedCounts();
            }

            trainer.terminator = new Terminator();
            trainer.maxNumLinearRegUpdates = maxNumLinearRegUpdates;
            return trainer;
        }
    }
}
