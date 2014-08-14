package edu.neu.ccs.pyramid.classification.boosting.lktb;

import edu.neu.ccs.pyramid.dataset.ClfDataSet;
import edu.neu.ccs.pyramid.dataset.DataSet;
import edu.neu.ccs.pyramid.dataset.FeatureRow;
import edu.neu.ccs.pyramid.regression.regression_tree.LeafOutputCalculator;
import edu.neu.ccs.pyramid.regression.regression_tree.RegTreeConfig;
import edu.neu.ccs.pyramid.regression.regression_tree.RegTreeTrainer;
import edu.neu.ccs.pyramid.regression.regression_tree.RegressionTree;
import edu.neu.ccs.pyramid.util.MathUtil;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

import java.util.Arrays;
import java.util.List;
import java.util.stream.IntStream;

/**
 * Created by chengli on 8/14/14.
 */
class LKTBTrainer {
    private static final Logger logger = LogManager.getLogger();
    /**
     * F_k(x), used to speed up training. stagedScore.[k][i] = F_k(x_i)
     */
    private double[][] stagedScore;
    /**
     * y_ik, classLabels[k][i]=y_ik
     */
    private int[][] classLabels;
    /**
     * p_k(x) classProbabilities[k][i] = p_k(x_i)
     */
    private double[][] classProbabilities;

    private LKTBConfig lktbConfig;

    /**
     * actually negative gradients, to be fit by the tree
     * classGradients[k]= gradients for class k
     */
    private double[][] classGradients;


    /**
     * when setting up a config in LKTB, also set up a trainer
     * @param lktbConfig
     */
    LKTBTrainer(LKTBConfig lktbConfig, List<List<RegressionTree>> trees){
        this.lktbConfig = lktbConfig;
        int numClasses = lktbConfig.getNumClasses();
        ClfDataSet dataSet= lktbConfig.getDataSet();
        int numDataPoints = dataSet.getNumDataPoints();
        this.stagedScore = new double[numClasses][numDataPoints];
        this.classLabels = new int[numClasses][numDataPoints];
        int[] trueLabels = dataSet.getLabels();
        for (int i=0;i<numDataPoints;i++){
            int label = trueLabels[i];
            this.classLabels[label][i] = 1;
        }
        this.initStagedScores(trees);
        this.classProbabilities = new double[numClasses][numDataPoints];
        this.updateClassProbs();
        this.classGradients = new double[numClasses][numDataPoints];
    }

    /**
     * parallel by classes
     * calculate gradient vectors for all classes, store them
     */
    void calGradients(){
        int numClasses = this.lktbConfig.getNumClasses();
        IntStream.range(0, numClasses).parallel()
                .forEach(this::calGradient);
    }

    double[] getGradient(int k){
        return this.classGradients[k];
    }

    /**
     * sum scores up
     * @param trees
     */
    private void initStagedScores(List<List<RegressionTree>> trees){
        int numClasses = this.lktbConfig.getNumClasses();
        ClfDataSet dataSet= this.lktbConfig.getDataSet();
        int numDataPoints = dataSet.getNumDataPoints();
        this.stagedScore = new double[numClasses][numDataPoints];
        for (int k=0;k<numClasses;k++){
            for (RegressionTree regressionTree: trees.get(k)){
                this.updateStagedScores(regressionTree,k);
            }
        }
    }

    /**
     * calculate gradient vector for class k, store it
     * @param k class index
     * @return pseudo response vector for class k
     */
    private void calGradient(int k){
        ClfDataSet dataSet= this.lktbConfig.getDataSet();
        int numDataPoints = dataSet.getNumDataPoints();
        int[] labelsClassK = this.classLabels[k];
        double[] probsClassK = this.classProbabilities[k];
        double[] gradient = this.classGradients[k];
        for (int i=0;i<numDataPoints;i++){
            gradient[i] = labelsClassK[i] - probsClassK[i];
        }
    }


    /**
     * use stagedScore to update probabilities
     * numerically unstable if calculated directly
     * probability = exp(log(nominator)-log(denominator))
     */
    private void updateClassProb(int i){
        int numClasses = this.lktbConfig.getNumClasses();
        double[] scores = new double[numClasses];

        for (int k=0;k<numClasses;k++){
            scores[k] = this.stagedScore[k][i];
        }
        double logDenominator = MathUtil.logSumExp(scores);
        if (logger.isDebugEnabled()){
            logger.debug("logDenominator for data point "+i+" with scores  = "+ Arrays.toString(scores)
                    +" ="+logDenominator+", label = "+lktbConfig.getDataSet().getLabels()[i]);
        }
        for (int k=0;k<numClasses;k++){
            double logNominator = this.stagedScore[k][i];
            double pro = Math.exp(logNominator-logDenominator);
            this.classProbabilities[k][i] = pro;
            if (Double.isNaN(pro)){
                throw new RuntimeException("pro=NaN, logNominator = "
                        +logNominator+", logDenominator="+logDenominator+
                        ", scores = "+Arrays.toString(scores));

            }
        }
    }

    /**
     * parallel by data points
     * update stagedScore of class k
     * @param regressionTree
     * @param k
     */
    void updateStagedScores(RegressionTree regressionTree, int k){
        ClfDataSet dataSet= this.lktbConfig.getDataSet();
        int numDataPoints = dataSet.getNumDataPoints();
        IntStream.range(0, numDataPoints).parallel()
                .forEach(dataIndex -> this.updateStagedScore(regressionTree,k,dataIndex));
    }

    /**
     * update one score
     * @param regressionTree
     * @param k class index
     * @param dataIndex
     */
    private void updateStagedScore(RegressionTree regressionTree, int k,
                                   int dataIndex){
        DataSet dataSet= this.lktbConfig.getDataSet();
        FeatureRow featureRow = dataSet.getFeatureRow(dataIndex);
        double prediction = regressionTree.predict(featureRow);
        this.stagedScore[k][dataIndex] += prediction;
    }

    /**
     * use stagedScore to update probabilities
     * parallel by data
     */
    void updateClassProbs(){
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
    RegressionTree fitClassK(int k) throws Exception{
        double[] pseudoResponse = this.classGradients[k];
        int numClasses = this.lktbConfig.getNumClasses();
        double learningRate = this.lktbConfig.getLearningRate();

        LeafOutputCalculator leafOutputCalculator = dataAppearance -> {
            double nominator = 0;
            double denominator = 0;
            for (int dataIndex : dataAppearance) {
                double label = pseudoResponse[dataIndex];
                nominator += label;
                denominator += Math.abs(label) * (1 - Math.abs(label));
            }
            double out;
            if (denominator == 0) {
                out = 0;
            } else {
                out = ((numClasses - 1) * nominator) / (numClasses * denominator);
            }
            if (Double.isNaN(out)) {
                throw new RuntimeException("leaf value is NaN");
            }
            out *= learningRate;
            return out;
        };

        RegTreeConfig regTreeConfig = new RegTreeConfig();
        regTreeConfig.setMaxNumLeaves(this.lktbConfig.getNumLeaves());
        regTreeConfig.setMinDataPerLeaf(this.lktbConfig.getMinDataPerLeaf());
        regTreeConfig.setActiveDataPoints(this.lktbConfig.getActiveDataPoints());
        regTreeConfig.setActiveFeatures(this.lktbConfig.getActiveFeatures());

        RegressionTree regressionTree = RegTreeTrainer.fit(regTreeConfig,
                this.lktbConfig.getDataSet(),
                pseudoResponse,
                leafOutputCalculator);
        return regressionTree;
    }

    void setActiveFeatures(int[] activeFeatures) {
        this.lktbConfig.setActiveFeatures(activeFeatures);
    }

    void setActiveDataPoints(int[] activeDataPoints) {
        this.lktbConfig.setActiveDataPoints(activeDataPoints);
    }

}
