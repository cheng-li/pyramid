package edu.neu.ccs.pyramid.multilabel_classification.hmlgb;

import edu.neu.ccs.pyramid.dataset.*;
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
     * F_k(x), used to speed up training.
     */
    private ScoreMatrix scoreMatrix;
    /**
     * [i][a]=prob of assignment a for x_i
     */
    private double[][] assignmentProbabilityMatrix;
    /**
     * gradients for maximum likelihood estimation, to be fit by the tree
     * gradientMatrix[k]= gradients for class k
     * store class first to ensure fast access of gradient
     */
    private GradientMatrix gradientMatrix;
    private ProbabilityMatrix probabilityMatrix;


    public HMLGBTrainer(HMLGBConfig config,
                        List<List<Regressor>> regressors,
                        List<MultiLabel> assignments) {
        this.config = config;
        this.assignments = assignments;
        MultiLabelClfDataSet dataSet = config.getDataSet();
        int numClasses = dataSet.getNumClasses();
        int numDataPoints = dataSet.getNumDataPoints();
        int numAssignments = this.assignments.size();
        this.scoreMatrix = new ScoreMatrix(numDataPoints,numClasses);
        this.initStagedClassScoreMatrix(regressors);
        this.assignmentProbabilityMatrix = new double[numDataPoints][numAssignments];
        this.updateAssignmentProbMatrix();
        this.gradientMatrix = new GradientMatrix(numDataPoints,numClasses, GradientMatrix.Objective.MAXIMIZE);
        this.probabilityMatrix = new ProbabilityMatrix(numDataPoints,numClasses);
        this.updateProbabilityMatrix();
    }

    double[] getGradients(int k){
        return this.gradientMatrix.getGradientsForClass(k);
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
     * update scoreMatrix of class k
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
        this.scoreMatrix.increment(dataIndex,k,prediction);
    }

    /**
     * use scoreMatrix to update probabilities
     * parallel by data
     */
    void updateAssignmentProbMatrix(){
        int numDataPoints = this.config.getDataSet().getNumDataPoints();
        IntStream.range(0,numDataPoints).parallel()
                .forEach(this::updateAssignmentProbs);
    }

    /**
     * use scoreMatrix to update probabilities
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
        double[] scores = this.scoreMatrix.getScoresForData(dataPoint);
        for (Integer label : assignment.getMatchedLabels()){
            score += scores[label];
        }
        return score;
    }

    /**
     * marginalized probability for each class
     * loop over all assignment probabilities
     * @param dataPoint
     * @return
     */
    private void updateClassProbs(int dataPoint){
        double[] assignmentProbs = this.assignmentProbabilityMatrix[dataPoint];
        int numAssignments = assignments.size();
        int numClasses = this.config.getDataSet().getNumClasses();
        //reset
        for (int k=0;k<numClasses;k++){
            this.probabilityMatrix.setProbability(dataPoint,k,0);
        }
        for (int a=0;a<numAssignments;a++){
            MultiLabel assignment = assignments.get(a);
            double prob = assignmentProbs[a];
            for (Integer label:assignment.getMatchedLabels()){
                this.probabilityMatrix.increment(dataPoint,label,prob);
            }
        }
    }


    void updateProbabilityMatrix(){
        int numDataPoints = this.config.getDataSet().getNumDataPoints();
        IntStream.range(0,numDataPoints).parallel()
                .forEach(this::updateClassProbs);
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
    RegressionTree fitClassK(int k){
        double[] gradients = gradientMatrix.getGradientsForClass(k);
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
