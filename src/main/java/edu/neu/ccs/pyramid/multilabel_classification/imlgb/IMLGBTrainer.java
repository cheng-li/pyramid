package edu.neu.ccs.pyramid.multilabel_classification.imlgb;

import edu.neu.ccs.pyramid.dataset.*;
import edu.neu.ccs.pyramid.multilabel_classification.MLPriorProbClassifier;
import edu.neu.ccs.pyramid.regression.ConstantRegressor;
import edu.neu.ccs.pyramid.regression.Regressor;
import edu.neu.ccs.pyramid.regression.regression_tree.LeafOutputCalculator;
import edu.neu.ccs.pyramid.regression.regression_tree.RegTreeConfig;
import edu.neu.ccs.pyramid.regression.regression_tree.RegTreeTrainer;
import edu.neu.ccs.pyramid.regression.regression_tree.RegressionTree;
import edu.neu.ccs.pyramid.util.MathUtil;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.apache.mahout.math.Vector;

import java.util.Arrays;
import java.util.List;
import java.util.stream.IntStream;

/**
 * Created by chengli on 10/8/14.
 */
public class IMLGBTrainer {
    private static final Logger logger = LogManager.getLogger();
    private IMLGBConfig config;

    /**
     * F_k(x), used to speed up training. scoreMatrix.[i][k] = F_k(x_i)
     */
    private ScoreMatrix scoreMatrix;

    /**
     * gradients for maximum likelihood estimation, to be fit by the tree
     * gradientMatrix[k]= gradients for class k
     * store class first to ensure fast access of gradient
     */
    private GradientMatrix gradientMatrix;
    private ProbabilityMatrix probabilityMatrix;
    private IMLGradientBoosting boosting;


    public IMLGBTrainer(IMLGBConfig config,
                        IMLGradientBoosting boosting) {
        if (config.getDataSet().getNumClasses()!=boosting.getNumClasses()){
            throw new IllegalArgumentException("config.getDataSet().getNumClasses()!=boosting.getNumClasses()");
        }
        this.config = config;
        this.boosting = boosting;
        MultiLabelClfDataSet dataSet = config.getDataSet();
        boosting.setFeatureList(dataSet.getFeatureList());
        boosting.setLabelTranslator(dataSet.getLabelTranslator());
        int numClasses = dataSet.getNumClasses();
        int numDataPoints = dataSet.getNumDataPoints();
        this.scoreMatrix = new ScoreMatrix(numDataPoints,numClasses);
        if (config.usePrior() && boosting.getRegressors(0).size()==0){
            this.setPriorProbs(dataSet);
        }
        this.initStagedClassScoreMatrix(boosting);
        this.probabilityMatrix = new ProbabilityMatrix(numDataPoints,numClasses);
        this.updateProbabilityMatrix();
        this.gradientMatrix = new GradientMatrix(numDataPoints,numClasses, GradientMatrix.Objective.MAXIMIZE);
        this.updateClassGradientMatrix();
    }

    public void iterate(){
        for (int k=0;k<this.boosting.getNumClasses();k++){
            /**
             * parallel by feature
             */
            Regressor regressor = this.fitClassK(k);
            this.boosting.addRegressor(regressor, k);
            /**
             * parallel by data
             */
            this.updateStagedClassScores(regressor,k);
        }
        this.updateProbabilityMatrix();
        this.updateClassGradientMatrix();
    }

    public ScoreMatrix getScoreMatrix() {
        return scoreMatrix;
    }

    public GradientMatrix getGradientMatrix() {
        return gradientMatrix;
    }

    public ProbabilityMatrix getProbabilityMatrix() {
        return probabilityMatrix;
    }

    public void setActiveFeatures(int[] activeFeatures) {
        this.config.setActiveFeatures(activeFeatures);
    }

    public void setActiveDataPoints(int[] activeDataPoints) {
        this.config.setActiveDataPoints(activeDataPoints);
    }

    //========================== PRIVATE ============================
    /**
     * not sure whether this is good for performance
     * start with prior probabilities
     * should be called before setTrainConfig
     * @param probs
     */
    private void setPriorProbs(double[] probs){
        if (probs.length!=this.boosting.getNumClasses()){
            throw new IllegalArgumentException("probs.length!=this.numClasses");
        }
        double average = Arrays.stream(probs).map(Math::log).average().getAsDouble();
        for (int k=0;k<this.boosting.getNumClasses();k++){
            double score = Math.log(probs[k] - average);
            Regressor constant = new ConstantRegressor(score);
            this.boosting.addRegressor(constant, k);
        }
    }

    /**
     * not sure whether this is good for performance
     * start with prior probabilities
     * should be called before setTrainConfig
     */
    private void setPriorProbs(MultiLabelClfDataSet dataSet){
        MLPriorProbClassifier priorProbClassifier = new MLPriorProbClassifier(dataSet.getNumClasses());
        priorProbClassifier.fit(dataSet);
        double[] probs = priorProbClassifier.getClassProbs();
        this.setPriorProbs(probs);
    }

    private void initStagedClassScoreMatrix(IMLGradientBoosting boosting){
        int numClasses = this.config.getDataSet().getNumClasses();
        for (int k=0;k<numClasses;k++){
            for (Regressor regressor: boosting.getRegressors(k)){
                this.updateStagedClassScores(regressor, k);
            }
        }
    }

    /**
     * parallel by data points
     * update scoreMatrix of class k
     * @param regressor
     * @param k
     */
    private void updateStagedClassScores(Regressor regressor, int k){
        DataSet dataSet= this.config.getDataSet();
        int numDataPoints = dataSet.getNumDataPoints();
        IntStream.range(0, numDataPoints).parallel()
                .forEach(dataIndex -> this.updateStagedClassScore(regressor, k, dataIndex));
    }

    /**
     * update one score
     * @param regressor
     * @param k class index
     * @param dataIndex
     */
    private void updateStagedClassScore(Regressor regressor, int k,
                                        int dataIndex){
        DataSet dataSet= this.config.getDataSet();
        Vector vector = dataSet.getRow(dataIndex);
        double prediction = regressor.predict(vector);
        this.scoreMatrix.increment(dataIndex,k,prediction);
    }

    private void updateProbabilityMatrix(){
        DataSet dataSet= this.config.getDataSet();
        int numDataPoints = dataSet.getNumDataPoints();
        IntStream.range(0, numDataPoints).parallel()
                .forEach(this::updateClassProbs);
    }

    /**
     * probability for each class
     * @param dataPoint
     * @return
     */
    private void updateClassProbs(int dataPoint){
        int numClasses = this.config.getDataSet().getNumClasses();
        for (int k=0;k<numClasses;k++){
            double prob = this.calClassProb(dataPoint, k);
            this.probabilityMatrix.setProbability(dataPoint,k,prob);
        }
    }

    private double calClassProb(int dataPoint, int k){
        double score = this.scoreMatrix.getScoresForData(dataPoint)[k];
        double logNumerator = score;
        double[] scores = new double[2];
        scores[0] = 0;
        scores[1] = score;
        double logDenominator = MathUtil.logSumExp(scores);
        double pro = Math.exp(logNumerator-logDenominator);
        return pro;
    }

    private void updateClassGradientMatrix(){
        int numDataPoints = this.config.getDataSet().getNumDataPoints();
        IntStream.range(0,numDataPoints).parallel()
                .forEach(this::updateClassGradients);
    }

    private void updateClassGradients(int dataPoint){
        int numClasses = this.config.getDataSet().getNumClasses();
        MultiLabel multiLabel = this.config.getDataSet().getMultiLabels()[dataPoint];
        //just use as a local variable
        //no need to store all in a matrix
        double[] classProbs = this.probabilityMatrix.getProbabilitiesForData(dataPoint);
        for (int k=0;k<numClasses;k++){
            double gradient = 0;
            if (multiLabel.matchClass(k)){
                gradient = 1-classProbs[k];
            } else {
                gradient = 0-classProbs[k];
            }
            this.gradientMatrix.setGradient(dataPoint,k,gradient);
        }
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
        double[] gradients = this.gradientMatrix.getGradientsForClass(k);
        int numClasses = this.config.getDataSet().getNumClasses();
        double learningRate = this.config.getLearningRate();

        LeafOutputCalculator leafOutputCalculator = probabilities -> {
            double numerator = 0;
            double denominator = 0;
            for (int i=0;i<probabilities.length;i++) {
                double label = gradients[i];
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
            if (out>2){
                out=2;
            }
            if (out<-2){
                out=-2;
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

        RegTreeConfig regTreeConfig = new RegTreeConfig();
        regTreeConfig.setMaxNumLeaves(this.config.getNumLeaves());
        regTreeConfig.setMinDataPerLeaf(this.config.getMinDataPerLeaf());

        regTreeConfig.setNumSplitIntervals(this.config.getNumSplitIntervals());

        RegressionTree regressionTree = RegTreeTrainer.fit(regTreeConfig,
                this.config.getDataSet(),
                gradients,
                leafOutputCalculator);
        return regressionTree;
    }



}
