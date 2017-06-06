package edu.neu.ccs.pyramid.multilabel_classification;

import edu.neu.ccs.pyramid.classification.ClassProbability;
import edu.neu.ccs.pyramid.dataset.LabelTranslator;
import edu.neu.ccs.pyramid.dataset.MultiLabel;
import edu.neu.ccs.pyramid.regression.ClassScoreCalculation;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.stream.Collectors;

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
    private List<ClassRankInfo> predictedRanking;
    private List<ClassScoreCalculation> classScoreCalculations;
    private List<LabelSetProbInfo> predictedLabelSetRanking;


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

    public List<ClassRankInfo> getPredictedRanking() {
        return predictedRanking;
    }

    public void setPredictedRanking(List<ClassRankInfo> predictedRanking) {
        Comparator<ClassRankInfo> comparator = Comparator.comparing(ClassRankInfo::getProb);
        this.predictedRanking = predictedRanking.stream().sorted(comparator.reversed()).collect(Collectors.toList());
    }

    public List<LabelSetProbInfo> getPredictedLabelSetRanking() {
        return predictedLabelSetRanking;
    }

    public void setPredictedLabelSetRanking(List<LabelSetProbInfo> predictedLabelSetRanking) {
        this.predictedLabelSetRanking = predictedLabelSetRanking;
    }

    public static class ClassRankInfo{
        private int classIndex;
        private String className;
        private double prob;

        public int getClassIndex() {
            return classIndex;
        }

        public void setClassIndex(int classIndex) {
            this.classIndex = classIndex;
        }

        public String getClassName() {
            return className;
        }

        public void setClassName(String className) {
            this.className = className;
        }

        public double getProb() {
            return prob;
        }

        public void setProb(double prob) {
            this.prob = prob;
        }
    }


    public static class LabelSetProbInfo {
        private List<Integer> internalLabels;
        private List<String> labels;
        private double probability;

        public LabelSetProbInfo(MultiLabel multiLabel, double probability, LabelTranslator labelTranslator) {
            this.probability = probability;
            this.internalLabels = multiLabel.getMatchedLabelsOrdered();
            this.labels = new ArrayList<>();
            for (int label: internalLabels){
                labels.add(labelTranslator.toExtLabel(label));
            }
        }

        public List<Integer> getInternalLabels() {
            return internalLabels;
        }

        public List<String> getLabels() {
            return labels;
        }

        public double getProbability() {
            return probability;
        }

        public void setInternalLabels(List<Integer> internalLabels) {
            this.internalLabels = internalLabels;
        }

        public void setLabels(List<String> labels) {
            this.labels = labels;
        }

        public void setProbability(double probability) {
            this.probability = probability;
        }
    }

}
