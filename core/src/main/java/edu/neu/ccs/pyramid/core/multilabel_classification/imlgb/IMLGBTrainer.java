package edu.neu.ccs.pyramid.core.multilabel_classification.imlgb;

import edu.neu.ccs.pyramid.core.dataset.*;
import edu.neu.ccs.pyramid.core.multilabel_classification.MLPriorProbClassifier;
import edu.neu.ccs.pyramid.core.regression.ConstantRegressor;
import edu.neu.ccs.pyramid.core.regression.Regressor;
import edu.neu.ccs.pyramid.core.regression.regression_tree.*;
import edu.neu.ccs.pyramid.core.util.MathUtil;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.apache.mahout.math.Vector;

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
//    private FloatGradientMatrix gradientMatrix;
    private IMLGradientBoosting boosting;

    private boolean[] shouldStop;


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
        List<MultiLabel> assignments = DataSetUtil.gatherMultiLabels(dataSet);
        boosting.setAssignments(assignments);
        this.shouldStop = new boolean[numClasses];
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

    public void iterate(){
        for (int k=0;k<this.boosting.getNumClasses();k++){

            if (!shouldStop[k]){
                if (logger.isDebugEnabled()){
                    logger.debug("updating class "+k);
                }
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
        return IntStream.range(0, this.config.getDataSet().getNumDataPoints()).parallel()
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
    private RegressionTree fitClassK(int k){

        double[] gradients = computeGradientForClass(k);
        int numClasses = this.config.getDataSet().getNumClasses();
        double learningRate = this.config.getLearningRate();


        LeafOutputCalculator leafOutputCalculator = new AverageOutputCalculator();

        RegTreeConfig regTreeConfig = new RegTreeConfig();
        regTreeConfig.setMaxNumLeaves(this.config.getNumLeaves());
        regTreeConfig.setMinDataPerLeaf(this.config.getMinDataPerLeaf());

        regTreeConfig.setNumSplitIntervals(this.config.getNumSplitIntervals());

        RegressionTree regressionTree = RegTreeTrainer.fit(regTreeConfig,
                this.config.getDataSet(),
                gradients,
                leafOutputCalculator);
        regressionTree.shrink(learningRate);
        return regressionTree;
    }



}
