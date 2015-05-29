package edu.neu.ccs.pyramid.classification.boosting.lktb;

import edu.neu.ccs.pyramid.classification.PriorProbClassifier;
import edu.neu.ccs.pyramid.classification.logistic_regression.LogisticRegression;
import edu.neu.ccs.pyramid.dataset.*;
import edu.neu.ccs.pyramid.regression.ConstantRegressor;
import edu.neu.ccs.pyramid.regression.Regressor;
import edu.neu.ccs.pyramid.regression.linear_regression.LinearRegression;
import edu.neu.ccs.pyramid.regression.probabilistic_regression_tree.ProbRegStump;
import edu.neu.ccs.pyramid.regression.probabilistic_regression_tree.ProbRegStumpTrainer;
import edu.neu.ccs.pyramid.regression.regression_tree.*;
import edu.neu.ccs.pyramid.util.MathUtil;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.apache.mahout.math.Vector;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.stream.IntStream;

/**
 * Created by chengli on 8/14/14.
 */
public class LKTBTrainer {
    private static final Logger logger = LogManager.getLogger();
    /**
     * F_k(x), used to speed up training.
     */
    private ScoreMatrix scoreMatrix;

    /**
     * p_k(x)
     */
    private ProbabilityMatrix probabilityMatrix;

    private LKTBConfig lktbConfig;

    /**
     * actually negative gradients, to be fit by the tree
     */
    private GradientMatrix gradientMatrix;
    private LKTreeBoost lkTreeBoost;


    public LKTBTrainer(LKTBConfig lktbConfig, LKTreeBoost lkTreeBoost){
        if (lktbConfig.getDataSet().getNumClasses()!=lkTreeBoost.getNumClasses()){
            throw new IllegalArgumentException("lktbConfig.getDataSet().getNumClasses()!=lkTreeBoost.getNumClasses()");
        }
        this.lktbConfig = lktbConfig;
        this.lkTreeBoost = lkTreeBoost;
        int numClasses = lkTreeBoost.getNumClasses();
        ClfDataSet dataSet= lktbConfig.getDataSet();
        lkTreeBoost.setFeatureList(dataSet.getFeatureList());
        lkTreeBoost.setLabelTranslator(dataSet.getLabelTranslator());
        int numDataPoints = dataSet.getNumDataPoints();
        this.scoreMatrix = new ScoreMatrix(numDataPoints,numClasses);
        this.initStagedScores();
        this.probabilityMatrix = new ProbabilityMatrix(numDataPoints,numClasses);
        this.updateProbabilityMatrix();
        this.gradientMatrix = new GradientMatrix(numDataPoints,numClasses, GradientMatrix.Objective.MAXIMIZE);
        this.updateGradientMatrix();
    }

    public void addRegressors(List<Regressor> regressors){
        int numClasses = lkTreeBoost.getNumClasses();
        if (regressors.size()!=numClasses){
            throw new IllegalArgumentException("regressors.size()!=numClasses");
        }
        for (int k=0;k<numClasses;k++){
            Regressor regressor = regressors.get(k);
            lkTreeBoost.addRegressor(regressor, k);
            /**
             * parallel by data
             */
            updateStagedScores(regressor, k);
        }

        /**
         * parallel by data
         */
        updateProbabilityMatrix();
        updateGradientMatrix();
    }

    public void addPriorRegressors(){
        PriorProbClassifier priorProbClassifier = new PriorProbClassifier(this.lkTreeBoost.getNumClasses());
        priorProbClassifier.fit(this.lktbConfig.getDataSet());
        double[] probs = priorProbClassifier.getClassProbs();
        double average = Arrays.stream(probs).map(Math::log).average().getAsDouble();
        List<Regressor> regressors = new ArrayList<>();
        for (int k=0;k<this.lkTreeBoost.getNumClasses();k++){
            double score = Math.log(probs[k] - average);
            Regressor constant = new ConstantRegressor(score);
            regressors.add(constant);
        }
        addRegressors(regressors);
    }

    public void addLogisticRegression(LogisticRegression logisticRegression){
        if (!lkTreeBoost.getRegressors(0).isEmpty()){
            throw new RuntimeException("adding logistic regression to non-empty model");
        }

        List<Regressor> regressors = new ArrayList<>();
        int numClasses = lkTreeBoost.getNumClasses();
        for (int k=0;k<numClasses;k++){
            //todo decide ratio
            Vector weightVector = logisticRegression.getWeights().getWeightsForClass(k).times(1);
            LinearRegression linearRegression = new LinearRegression(logisticRegression.getNumFeatures(),weightVector);
            regressors.add(linearRegression);
        }
        addRegressors(regressors);
    }


    /**
     * by default, add trees in each iteration
     */
    public void iterate(){
        int numClasses = lkTreeBoost.getNumClasses();
        List<Regressor> regressors = new ArrayList<>();
        for (int k=0;k<numClasses;k++){
            /**
             * parallel by feature
             */
            Regressor regressor = fitClassK(k);
            regressors.add(regressor);
        }
        addRegressors(regressors);
    }

    public void setActiveFeatures(int[] activeFeatures) {
        this.lktbConfig.setActiveFeatures(activeFeatures);
    }

    public void setActiveDataPoints(int[] activeDataPoints) {
        this.lktbConfig.setActiveDataPoints(activeDataPoints);
    }

    public GradientMatrix getGradientMatrix() {
        return gradientMatrix;
    }

    public ProbabilityMatrix getProbabilityMatrix() {
        return probabilityMatrix;
    }



    //======================== PRIVATE ===============================================


    /**
     * parallel by classes
     * calculate gradient vectors for all classes, store them
     */
    private void updateGradientMatrix(){
        int numDataPoints = this.lktbConfig.getDataSet().getNumDataPoints();
        IntStream.range(0, numDataPoints).parallel()
                .forEach(this::updateClassGradients);
    }


    private void initStagedScores(){
        int numClasses = this.lkTreeBoost.getNumClasses();
        for (int k=0;k<numClasses;k++){
            for (Regressor regressor: lkTreeBoost.getRegressors(k)){
                this.updateStagedScores(regressor,k);
            }
        }
    }

    private void updateClassGradients(int dataPoint){
        int numClasses = this.lkTreeBoost.getNumClasses();
        int label = this.lktbConfig.getDataSet().getLabels()[dataPoint];
        double[] probs = this.probabilityMatrix.getProbabilitiesForData(dataPoint);
        for (int k=0;k<numClasses;k++){
            double gradient;
            if (label==k){
                gradient = 1-probs[k];
            } else {
                gradient = 0-probs[k];
            }
            this.gradientMatrix.setGradient(dataPoint,k,gradient);
        }
    }


    /**
     * use scoreMatrix to update probabilities
     * numerically unstable if calculated directly
     * probability = exp(log(numerator)-log(denominator))
     */
    private void updateClassProb(int i){
        int numClasses = this.lkTreeBoost.getNumClasses();
        double[] scores = scoreMatrix.getScoresForData(i);

        double logDenominator = MathUtil.logSumExp(scores);
//        if (logger.isDebugEnabled()){
//            logger.debug("logDenominator for data point "+i+" with scores  = "+ Arrays.toString(scores)
//                    +" ="+logDenominator+", label = "+lktbConfig.getDataSet().getLabels()[i]);
//        }
        for (int k=0;k<numClasses;k++){
            double logNumerator = scores[k];
            double pro = Math.exp(logNumerator-logDenominator);
            this.probabilityMatrix.setProbability(i,k,pro);
            if (Double.isNaN(pro)){
                throw new RuntimeException("pro=NaN, logNumerator = "
                        +logNumerator+", logDenominator="+logDenominator+
                        ", scores = "+Arrays.toString(scores));

            }
        }
    }

    /**
     * parallel by data points
     * update scoreMatrix of class k
     * @param regressor
     * @param k
     */
    private void updateStagedScores(Regressor regressor, int k){
        ClfDataSet dataSet= this.lktbConfig.getDataSet();
        int numDataPoints = dataSet.getNumDataPoints();
        IntStream.range(0, numDataPoints).parallel()
                .forEach(dataIndex -> this.updateStagedScore(regressor,k,dataIndex));
    }

    /**
     * update one score
     * @param regressor
     * @param k class index
     * @param dataIndex
     */
    private void updateStagedScore(Regressor regressor, int k,
                                   int dataIndex){
        DataSet dataSet= this.lktbConfig.getDataSet();
        Vector vector = dataSet.getRow(dataIndex);
        double prediction = regressor.predict(vector);
        this.scoreMatrix.increment(dataIndex,k,prediction);
    }

    /**
     * use scoreMatrix to update probabilities
     * parallel by data
     */
    private void updateProbabilityMatrix(){
        ClfDataSet dataSet= this.lktbConfig.getDataSet();
        int numDataPoints = dataSet.getNumDataPoints();
        IntStream.range(0,numDataPoints).parallel()
                .forEach(this::updateClassProb);
    }

    /**
     * parallel
     * find the best regression tree for class k
     * apply newton step and learning rate
     * @param k class index
     * @return regressionTreeLk, shrunk
     * @throws Exception
     */
    private RegressionTree fitClassK(int k){
        double[] pseudoResponse = this.gradientMatrix.getGradientsForClass(k);
        int numClasses = this.lkTreeBoost.getNumClasses();
        double learningRate = this.lktbConfig.getLearningRate();

        LeafOutputCalculator leafOutputCalculator = null;

        switch (lktbConfig.getLeafOutputType()){
            case NEWTON:
                leafOutputCalculator = probabilities -> {
                    double numerator = 0;
                    double denominator = 0;
                    for (int i=0;i<probabilities.length;i++) {
                        double label = pseudoResponse[i];
                        numerator += label*probabilities[i];
                        denominator += Math.abs(label) * (1 - Math.abs(label))*probabilities[i];
                    }
                    double out;
                    if (denominator == 0) {
                        out = 0;
                    } else {
                        out = ((numClasses - 1) * numerator) / (numClasses * denominator);
                    }
                    //protection from numerically unstable issue
                    //todo does the threshold matter?
                    if (out>1){
                        out=1;
                    }
                    if (out<-1){
                        out=-1;
                    }
                    if (Double.isNaN(out)) {
                        throw new RuntimeException("leaf value is NaN");
                    }
                    if (Double.isInfinite(out)){
                        throw new RuntimeException("leaf value is Infinite");
                    }
                    out *= learningRate;
                    return out;
                };
                break;
            case AVERAGE:
                leafOutputCalculator = new AverageOutputCalculator(pseudoResponse);
                break;
        }


        RegTreeConfig regTreeConfig = new RegTreeConfig();
        regTreeConfig.setMaxNumLeaves(this.lktbConfig.getNumLeaves());
        regTreeConfig.setMinDataPerLeaf(this.lktbConfig.getMinDataPerLeaf());
        regTreeConfig.setActiveDataPoints(this.lktbConfig.getActiveDataPoints());
        regTreeConfig.setActiveFeatures(this.lktbConfig.getActiveFeatures());
        regTreeConfig.setNumSplitIntervals(this.lktbConfig.getNumSplitIntervals());
        regTreeConfig.setRandomLevel(this.lktbConfig.getRandomLevel());


        RegressionTree regressionTree = RegTreeTrainer.fit(regTreeConfig,
                this.lktbConfig.getDataSet(),
                pseudoResponse,
                leafOutputCalculator);

        return regressionTree;
    }


//    private Regressor fitClassK(int k){
//        double[] pseudoResponse = this.gradientMatrix.getGradientsForClass(k);
//        double learningRate = this.lktbConfig.getLearningRate();
//
//        ProbRegStumpTrainer trainer = ProbRegStumpTrainer.getBuilder()
//                .setDataSet(lktbConfig.getDataSet())
//                .setLabels(pseudoResponse)
//                .setFeatureType(ProbRegStumpTrainer.FeatureType.FOLLOW_HARD_TREE_FEATURE)
//                .setLossType(ProbRegStumpTrainer.LossType.SquaredLossOfExpectation)
//                .build();
//
//        ProbRegStump tree = trainer.train();
//        tree.shrink(learningRate);
//        return tree;
//    }



}
