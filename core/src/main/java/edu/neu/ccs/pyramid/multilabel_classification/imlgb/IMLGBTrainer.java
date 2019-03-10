package edu.neu.ccs.pyramid.multilabel_classification.imlgb;

import edu.neu.ccs.pyramid.dataset.*;
import edu.neu.ccs.pyramid.multilabel_classification.MLPriorProbClassifier;
import edu.neu.ccs.pyramid.regression.ConstantRegressor;
import edu.neu.ccs.pyramid.regression.Regressor;
import edu.neu.ccs.pyramid.regression.regression_tree.*;
import edu.neu.ccs.pyramid.util.MathUtil;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.apache.mahout.math.Vector;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.stream.Collectors;
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
//    private FloatGradientMatrix gradientMatrix;
    private IMLGradientBoosting boosting;

    private boolean[] shouldStop;
    private double[] instanceWeights;


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
//        this.gradientMatrix = new FloatGradientMatrix(numDataPoints,numClasses, FloatGradientMatrix.Objective.MAXIMIZE);
//        List<MultiLabel> assignments = DataSetUtil.gatherMultiLabels(dataSet);
//        boosting.setAssignments(assignments);
        this.shouldStop = new boolean[numClasses];
        this.instanceWeights = new double[dataSet.getNumDataPoints()];
        Arrays.fill(instanceWeights,1.0);
    }

    public IMLGBTrainer(IMLGBConfig config,
                        IMLGradientBoosting boosting,
                        boolean[] shouldStop) {
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
        this.shouldStop = shouldStop;
        this.initStagedClassScoreMatrix(boosting, shouldStop);
        this.instanceWeights = new double[dataSet.getNumDataPoints()];
        Arrays.fill(instanceWeights,1.0);
    }

    public void setShouldStop(int classIndex){
        shouldStop[classIndex] = true;
        if (logger.isDebugEnabled()){
            logger.debug("class "+classIndex+" is set to stop");
        }
    }

    public boolean[] getShouldStop() {
        return shouldStop;
    }

    public void setInstanceWeights(double[] instanceWeights) {
        this.instanceWeights = instanceWeights;
    }

    public void iterate(){

        List<Integer> allFeatureIndices = IntStream.range(0, this.config.getDataSet().getNumFeatures()).boxed().collect(Collectors.toList());


        IntStream.range(0, this.boosting.getNumClasses()).parallel()
                .forEach(k->{
                    if (!shouldStop[k]){
                        if (logger.isDebugEnabled()){
                            logger.debug("updating class "+k);
                        }
                        Regressor regressor = this.fitClassK(k, allFeatureIndices,false);
                        this.boosting.addRegressor(regressor, k);

                        this.updateStagedClassScores(regressor,k);
                    }
                });
    }


    public void iterate(List<Integer>[] activeFeatureLists, boolean fullScan){
        if (fullScan){

            IntStream.range(0, this.boosting.getNumClasses()).parallel()
                .forEach(k->{
                    if (!shouldStop[k]){
                        if (logger.isDebugEnabled()){
                            logger.debug("updating class "+k);
                        }
                        Regressor regressor = this.fitClassK(k, activeFeatureLists[k], true);
                        this.boosting.addRegressor(regressor, k);

                        this.updateStagedClassScores(regressor,k);
                    }
                });

        }else{
            IntStream.range(0, this.boosting.getNumClasses()).parallel()
                .forEach(k->{
                    if (!shouldStop[k]){
                        if (logger.isDebugEnabled()){
                            logger.debug("updating class "+k);
                        }
                        Regressor regressor = this.fitClassK(k, activeFeatureLists[k], false);
                        this.boosting.addRegressor(regressor, k);

                        this.updateStagedClassScores(regressor,k);
                    }
                });
        }


    }


    public void iterateWithoutStagingScores(List<Integer>[] activeFeatureLists, boolean fullScan){
        if(fullScan){
            IntStream.range(0, this.boosting.getNumClasses()).parallel()
                    .forEach(k->{
                        if (!shouldStop[k]){
                            if (logger.isDebugEnabled()){
                                logger.debug("updating class "+k);
                            }
                            Regressor regressor = this.fitClassK(k, activeFeatureLists[k], true);
                            this.boosting.addRegressor(regressor, k);
                        }
                    });


        }else{
            IntStream.range(0, this.boosting.getNumClasses()).parallel()
                    .forEach(k->{
                        if (!shouldStop[k]){
                            if (logger.isDebugEnabled()){
                                logger.debug("updating class "+k);
                            }
                            Regressor regressor = this.fitClassK(k, activeFeatureLists[k], false);
                            this.boosting.addRegressor(regressor, k);
                        }
                    });

        }

    }




//    public void setActiveFeatures(int[] activeFeatures) {
//        this.config.setActiveFeatures(activeFeatures);
//    }
//
//    public void setActiveDataPoints(int[] activeDataPoints) {
//        this.config.setActiveDataPoints(activeDataPoints);
//    }

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

        for (int k=0;k<this.boosting.getNumClasses();k++){
            double score = MathUtil.inverseSigmoid(probs[k]);
            // we don't want the prior to be overly strong
            double soft = Math.sqrt(Math.abs(score));
            if (score<0){
                soft = -soft;
            }
            Regressor constant = new ConstantRegressor(soft);
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
        DataSet dataSet = config.getDataSet();
        IntStream.range(0, dataSet.getNumDataPoints()).parallel()
                .forEach(i->{
                    double[] classScores = boosting.predictClassScoresCachedInput(dataSet.getRow(i));
                    for (int k=0;k<boosting.getNumClasses();k++){
                        scoreMatrix.setScore(i,k,classScores[k]);
                    }
                });

//        int numClasses = this.config.getDataSet().getNumClasses();
//        IntStream.range(0, numClasses).parallel()
//        .forEach(k-> {
//                    for (Regressor regressor : boosting.getRegressors(k)) {
//                        this.updateStagedClassScores(regressor, k);
//                    }});


    }

    private void initStagedClassScoreMatrix(IMLGradientBoosting boosting, boolean[] shouldStop){
        DataSet dataSet = config.getDataSet();
        IntStream.range(0, dataSet.getNumDataPoints()).parallel()
                .forEach(i->{
                    double[] classScores = boosting.predictClassScoresCachedInput(dataSet.getRow(i), shouldStop);
                    for (int k=0;k<boosting.getNumClasses();k++){
                        if (!shouldStop[k]){
                            scoreMatrix.setScore(i,k,classScores[k]);
                        }
                    }
                });
    }



    /**
     * update scoreMatrix of class k
     * @param regressor
     * @param k
     */
    private void updateStagedClassScores(Regressor regressor, int k){
        DataSet dataSet= this.config.getDataSet();
        int numDataPoints = dataSet.getNumDataPoints();
        IntStream.range(0, numDataPoints)
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



    private double[] computeGradientForClass(int k){
        return IntStream.range(0, this.config.getDataSet().getNumDataPoints())
                .mapToDouble(i->computeGradient(k,i)).toArray();
    }

    private double computeGradient(int k, int dataPoint){
        MultiLabel multiLabel = this.config.getDataSet().getMultiLabels()[dataPoint];
        double classProb = this.calClassProb(dataPoint, k);
        double gradient;
        if (multiLabel.matchClass(k)){
            gradient = 1-classProb;
        } else {
            gradient = 0-classProb;
        }
        return gradient;
    }

    /**
     * parallel
     * find the best regression tree for class k
     * apply newton step and learning rate
     * @param k class index
     * @return regressionTreeLk, shrunk
     * @throws Exception
     */
    private RegressionTree fitClassK(int k, List<Integer> activeFeatures, boolean fullScan){
        if (fullScan){
            double[] gradients = computeGradientForClass(k);
            double learningRate = this.config.getLearningRate();


            LeafOutputCalculator leafOutputCalculator = new AverageOutputCalculator();

            RegTreeConfig regTreeConfig = new RegTreeConfig();
            regTreeConfig.setMaxNumLeaves(this.config.getNumLeaves());
            regTreeConfig.setMinDataPerLeaf(this.config.getMinDataPerLeaf());

            regTreeConfig.setNumSplitIntervals(this.config.getNumSplitIntervals());
            regTreeConfig.setParallel(false);
            regTreeConfig.setNumActiveFeatures(this.config.getNumActiveFeatures());

            RegressionTree regressionTree = ActiveRegTreeTrainer.fit(regTreeConfig,
                    this.config.getDataSet(),
                    gradients,
                    instanceWeights,
                    leafOutputCalculator,
                    activeFeatures,
                    true);
            regressionTree.shrink(learningRate);
            return regressionTree;

        }else{

            double[] gradients = computeGradientForClass(k);
            double learningRate = this.config.getLearningRate();


            LeafOutputCalculator leafOutputCalculator = new AverageOutputCalculator();

            RegTreeConfig regTreeConfig = new RegTreeConfig();
            regTreeConfig.setMaxNumLeaves(this.config.getNumLeaves());
            regTreeConfig.setMinDataPerLeaf(this.config.getMinDataPerLeaf());

            regTreeConfig.setNumSplitIntervals(this.config.getNumSplitIntervals());
            regTreeConfig.setParallel(false);
            regTreeConfig.setNumActiveFeatures(this.config.getNumActiveFeatures());

            RegressionTree regressionTree = ActiveRegTreeTrainer.fit(regTreeConfig,
                    this.config.getDataSet(),
                    gradients,
                    instanceWeights,
                    leafOutputCalculator,
                    activeFeatures,
                    false);
            regressionTree.shrink(learningRate);
            return regressionTree;
        }




    }



}
