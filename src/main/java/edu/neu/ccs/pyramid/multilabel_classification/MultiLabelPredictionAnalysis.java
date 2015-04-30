package edu.neu.ccs.pyramid.multilabel_classification;

import edu.neu.ccs.pyramid.classification.ClassProbability;
import edu.neu.ccs.pyramid.regression.ClassScoreCalculation;

import java.util.List;

/**
 * Created by chengli on 2/28/15.
 */
public class MultiLabelPredictionAnalysis {
    private int internalId;
    private String id;
    private List<Integer> internalLabels;
    private List<String> labels;
    private double probForTrueLabels;
    private List<Integer> internalPrediction;
    private List<String> prediction;
    private double probForPredictedLabels;
    private List<String> predictedRanking;
    private List<ClassScoreCalculation> classScoreCalculations;


    public MultiLabelPredictionAnalysis() {
    }

    public int getInternalId() {
        return internalId;
    }

    public void setInternalId(int internalId) {
        this.internalId = internalId;
    }

    public String getId() {
        return id;
    }

    public void setId(String id) {
        this.id = id;
    }

    public List<Integer> getInternalLabels() {
        return internalLabels;
    }

    public void setInternalLabels(List<Integer> internalLabels) {
        this.internalLabels = internalLabels;
    }

    public List<String> getLabels() {
        return labels;
    }

    public void setLabels(List<String> labels) {
        this.labels = labels;
    }

    public List<Integer> getInternalPrediction() {
        return internalPrediction;
    }

    public void setInternalPrediction(List<Integer> internalPrediction) {
        this.internalPrediction = internalPrediction;
    }

    public List<String> getPrediction() {
        return prediction;
    }

    public void setPrediction(List<String> prediction) {
        this.prediction = prediction;
    }

    public List<ClassScoreCalculation> getClassScoreCalculations() {
        return classScoreCalculations;
    }

    public void setClassScoreCalculations(List<ClassScoreCalculation> classScoreCalculations) {
        this.classScoreCalculations = classScoreCalculations;
    }

    public double getProbForTrueLabels() {
        return probForTrueLabels;
    }

    public void setProbForTrueLabels(double probForTrueLabels) {
        this.probForTrueLabels = probForTrueLabels;
    }

    public double getProbForPredictedLabels() {
        return probForPredictedLabels;
    }

    public void setProbForPredictedLabels(double probForPredictedLabels) {
        this.probForPredictedLabels = probForPredictedLabels;
    }

    public List<String> getPredictedRanking() {
        return predictedRanking;
    }

    public void setPredictedRanking(List<String> predictedRanking) {
        this.predictedRanking = predictedRanking;
    }
}
