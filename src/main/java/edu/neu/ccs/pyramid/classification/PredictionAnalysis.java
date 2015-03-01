package edu.neu.ccs.pyramid.classification;

import edu.neu.ccs.pyramid.regression.ClassScoreCalculation;

import java.util.List;

/**
 * Created by chengli on 2/28/15.
 */
public class PredictionAnalysis {
    private int internalId;
    private String id;
    private int internalLabel;
    private String label;
    private int internalPrediction;
    private String prediction;
    private List<ClassProbability> classProbabilities;
    private List<ClassScoreCalculation> classScoreCalculations;

    public PredictionAnalysis() {
    }

    public PredictionAnalysis setInternalId(int internalId) {
        this.internalId = internalId;
        return this;
    }

    public PredictionAnalysis setId(String id) {
        this.id = id;
        return this;
    }

    public PredictionAnalysis setInternalLabel(int internalLabel) {
        this.internalLabel = internalLabel;
        return this;
    }

    public PredictionAnalysis setLabel(String label) {
        this.label = label;
        return this;
    }

    public PredictionAnalysis setInternalPrediction(int internalPrediction) {
        this.internalPrediction = internalPrediction;
        return this;
    }

    public PredictionAnalysis setPrediction(String prediction) {
        this.prediction = prediction;
        return this;
    }

    public PredictionAnalysis setClassProbabilities(List<ClassProbability> classProbabilities) {
        this.classProbabilities = classProbabilities;
        return this;
    }

    public PredictionAnalysis setClassScoreCalculations(List<ClassScoreCalculation> classScoreCalculations) {
        this.classScoreCalculations = classScoreCalculations;
        return this;
    }

    public int getInternalId() {
        return internalId;
    }

    public String getId() {
        return id;
    }

    public int getInternalLabel() {
        return internalLabel;
    }

    public String getLabel() {
        return label;
    }

    public int getInternalPrediction() {
        return internalPrediction;
    }

    public String getPrediction() {
        return prediction;
    }

    public List<ClassProbability> getClassProbabilities() {
        return classProbabilities;
    }

    public List<ClassScoreCalculation> getClassScoreCalculations() {
        return classScoreCalculations;
    }
}
