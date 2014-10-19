package edu.neu.ccs.pyramid.multilabel_classification.hmlgb;

import edu.neu.ccs.pyramid.dataset.DataSet;
import edu.neu.ccs.pyramid.dataset.FeatureRow;
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


import java.util.List;

import java.util.stream.IntStream;

/**
 * Created by chengli on 9/27/14.
 */
public class HMLGBTrainer {
    private static final Logger logger = LogManager.getLogger();
    private HMLGBConfig config;
    /**
     * legal assignments of labels
     */
    private List<MultiLabel> assignments;
    /**
     * F_k(x), used to speed up training. stagedClassScoreMatrix.[i][k] = F_k(x_i)
     */
    private double[][] stagedClassScoreMatrix;
    /**
     * [i][a]=prob of assignment a for x_i
     */
    private double[][] assignmentProbabilityMatrix;
    /**
     * gradients for maximum likelihood estimation, to be fit by the tree
     * classGradientMatrix[k]= gradients for class k
     * store class first to ensure fast access of gradient
     */
    private double[][] classGradientMatrix;


    public HMLGBTrainer(HMLGBConfig config,
                        List<List<Regressor>> regressors,
                        List<MultiLabel> assignments) {
        this.config = config;
        this.assignments = assignments;
        MultiLabelClfDataSet dataSet = config.getDataSet();
        int numClasses = dataSet.getNumClasses();
        int numDataPoints = dataSet.getNumDataPoints();
        int numAssignments = this.assignments.size();
        this.stagedClassScoreMatrix = new double[numDataPoints][numClasses];
        this.initStagedClassScoreMatrix(regressors);
        this.assignmentProbabilityMatrix = new double[numDataPoints][numAssignments];
        this.updateAssignmentProbMatrix();
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
        FeatureRow featureRow = dataSet.getFeatureRow(dataIndex);
        double prediction = regressor.predict(featureRow);
        this.stagedClassScoreMatrix[dataIndex][k] += prediction;
    }

    /**
     * use stagedClassScoreMatrix to update probabilities
     * parallel by data
     */
    void updateAssignmentProbMatrix(){
        int numDataPoints = this.config.getDataSet().getNumDataPoints();
        IntStream.range(0,numDataPoints).parallel()
                .forEach(this::updateAssignmentProbs);
    }

    /**
     * use stagedClassScoreMatrix to update probabilities
     * numerically unstable if calculated directly
     * probability = exp(log(nominator)-log(denominator))
     */
    private void updateAssignmentProbs(int dataPoint){
        int numAssignments = assignments.size();
        double[] assignmentScores = new double[numAssignments];
        for (int a=0;a<numAssignments;a++){
            MultiLabel assignment = assignments.get(a);
            assignmentScores[a] = this.calAssignmentScores(dataPoint, assignment);
        }
        double logDenominator = MathUtil.logSumExp(assignmentScores);

        for (int a=0;a<numAssignments;a++){
            double logNominator = assignmentScores[a];
            double pro = Math.exp(logNominator-logDenominator);
            this.assignmentProbabilityMatrix[dataPoint][a]=pro;
        }
    }


    private double calAssignmentScores(int dataPoint, MultiLabel assignment){
        double score = 0;
        for (Integer label : assignment.getMatchedLabels()){
            score += this.stagedClassScoreMatrix[dataPoint][label];
        }
        return score;
    }

    /**
     * marginalized probability for each class
     * loop over all assignment probabilities
     * @param dataPoint
     * @return
     */
    private double[] calClassProbs(int dataPoint){
        double[] assignmentProbs = this.assignmentProbabilityMatrix[dataPoint];
        double[] classProbs = new double[this.config.getDataSet().getNumClasses()];
        int numAssignments = assignments.size();
        for (int a=0;a<numAssignments;a++){
            MultiLabel assignment = assignments.get(a);
            for (Integer label:assignment.getMatchedLabels()){
                classProbs[label] += assignmentProbs[label];
            }
        }
        return classProbs;
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
