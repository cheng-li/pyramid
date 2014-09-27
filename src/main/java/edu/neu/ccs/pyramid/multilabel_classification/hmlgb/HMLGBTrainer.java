package edu.neu.ccs.pyramid.multilabel_classification.hmlgb;

import edu.neu.ccs.pyramid.dataset.DataSet;
import edu.neu.ccs.pyramid.dataset.FeatureRow;
import edu.neu.ccs.pyramid.dataset.MultiLabel;
import edu.neu.ccs.pyramid.regression.Regressor;
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


    /**
     * sum scores up
     * @param regressors
     */
    private void initStagedClassScoreMatrix(List<List<Regressor>> regressors){
        int numClasses = this.config.getDataSet().getNumClasses();
        DataSet dataSet= this.config.getDataSet();
        int numDataPoints = dataSet.getNumDataPoints();
        this.stagedClassScoreMatrix = new double[numDataPoints][numClasses];
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
        List<MultiLabel> assignments = config.getAssignments();
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
        List<MultiLabel> assignments = config.getAssignments();
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


}
