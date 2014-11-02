package edu.neu.ccs.pyramid.multilabel_classification.imlgb;

import edu.neu.ccs.pyramid.dataset.DataSet;
import edu.neu.ccs.pyramid.dataset.MultiLabel;
import edu.neu.ccs.pyramid.dataset.MultiLabelClfDataSet;
import edu.neu.ccs.pyramid.regression.Regressor;
import edu.neu.ccs.pyramid.regression.regression_tree.LeafOutputCalculator;
import edu.neu.ccs.pyramid.regression.regression_tree.RegTreeConfig;
import edu.neu.ccs.pyramid.regression.regression_tree.RegTreeTrainer;
import edu.neu.ccs.pyramid.regression.regression_tree.RegressionTree;
import edu.neu.ccs.pyramid.util.MathUtil;
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
     * F_k(x), used to speed up training. stagedClassScoreMatrix.[i][k] = F_k(x_i)
     */
    private double[][] stagedClassScoreMatrix;

    /**
     * gradients for maximum likelihood estimation, to be fit by the tree
     * classGradientMatrix[k]= gradients for class k
     * store class first to ensure fast access of gradient
     */
    private double[][] classGradientMatrix;


    public IMLGBTrainer(IMLGBConfig config,
                        List<List<Regressor>> regressors) {
        this.config = config;
        MultiLabelClfDataSet dataSet = config.getDataSet();
        int numClasses = dataSet.getNumClasses();
        int numDataPoints = dataSet.getNumDataPoints();
        this.stagedClassScoreMatrix = new double[numDataPoints][numClasses];
        this.initStagedClassScoreMatrix(regressors);
        this.classGradientMatrix = new double[numClasses][numDataPoints];
    }

    double[] getGradients(int k){
        return this.classGradientMatrix[k];
    }

    /**
     * sum scores up
     * @param regressors
     */
    private void initStagedClassScoreMatrix(List<List<Regressor>> regressors){
        int numClasses = this.config.getDataSet().getNumClasses();
        for (int k=0;k<numClasses;k++){
            for (Regressor regressor: regressors.get(k)){
                this.updateStagedClassScores(regressor, k);
            }
        }
    }

    /**
     * parallel by data points
     * update stagedClassScoreMatrix of class k
     * @param regressor
     * @param k
     */
    void updateStagedClassScores(Regressor regressor, int k){
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
        this.stagedClassScoreMatrix[dataIndex][k] += prediction;
    }

    /**
     * probability for each class
     * @param dataPoint
     * @return
     */
    private double[] calClassProbs(int dataPoint){
        int numClasses = this.config.getDataSet().getNumClasses();
        double[] classProbs = new double[numClasses];
        for (int k=0;k<numClasses;k++){
            classProbs[k] = this.calClassProb(dataPoint,k);
        }
        return classProbs;
    }

    private double calClassProb(int dataPoint, int k){
        double score = this.stagedClassScoreMatrix[dataPoint][k];
        double logNominator = score;
        double[] scores = new double[2];
        scores[0] = 0;
        scores[1] = score;
        double logDenominator = MathUtil.logSumExp(scores);
        double pro = Math.exp(logNominator-logDenominator);
        return pro;
    }

    void updateClassGradientMatrix(){
        int numDataPoints = this.config.getDataSet().getNumDataPoints();
        IntStream.range(0,numDataPoints).parallel()
                .forEach(this::updateClassGradients);
    }

    private void updateClassGradients(int dataPoint){
        int numClasses = this.config.getDataSet().getNumClasses();
        MultiLabel multiLabel = this.config.getDataSet().getMultiLabels()[dataPoint];
        //just use as a local variable
        //no need to store all in a matrix
        double[] classProbs = this.calClassProbs(dataPoint);
        for (int k=0;k<numClasses;k++){
            double gradient = 0;
            if (multiLabel.matchClass(k)){
                gradient = 1-classProbs[k];
            } else {
                gradient = 0-classProbs[k];
            }
            this.classGradientMatrix[k][dataPoint] = gradient;
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
    RegressionTree fitClassK(int k){
        double[] gradients = this.classGradientMatrix[k];
        int numClasses = this.config.getDataSet().getNumClasses();
        double learningRate = this.config.getLearningRate();

        LeafOutputCalculator leafOutputCalculator = probabilities -> {
            double nominator = 0;
            double denominator = 0;
            for (int i=0;i<probabilities.length;i++) {
                double label = gradients[i];
                nominator += label*probabilities[i];
                denominator += Math.abs(label) * (1 - Math.abs(label))*probabilities[i];
            }
            double out;
            if (denominator == 0) {
                out = 0;
            } else {
                out = ((numClasses - 1) * nominator) / (numClasses * denominator);
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
        regTreeConfig.setActiveDataPoints(this.config.getActiveDataPoints());
        regTreeConfig.setActiveFeatures(this.config.getActiveFeatures());
        regTreeConfig.setNumSplitIntervals(this.config.getNumSplitIntervals());

        RegressionTree regressionTree = RegTreeTrainer.fit(regTreeConfig,
                this.config.getDataSet(),
                gradients,
                leafOutputCalculator);
        return regressionTree;
    }

    void setActiveFeatures(int[] activeFeatures) {
        this.config.setActiveFeatures(activeFeatures);
    }

    void setActiveDataPoints(int[] activeDataPoints) {
        this.config.setActiveDataPoints(activeDataPoints);
    }

}
