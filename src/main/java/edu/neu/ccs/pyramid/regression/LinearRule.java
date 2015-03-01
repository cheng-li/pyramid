package edu.neu.ccs.pyramid.regression;

/**
 * Created by chengli on 2/28/15.
 */
public class LinearRule implements Rule {
    private int featureIndex;
    private String featureName;
    private double weight;
    private double featureValue;
    private double score;

    public LinearRule(int featureIndex, String featureName,
                      double weight, double featureValue) {
        this.featureIndex = featureIndex;
        this.featureName = featureName;
        this.weight = weight;
        this.featureValue = featureValue;
        this.score = weight*featureValue;
    }

    public int getFeatureIndex() {
        return featureIndex;
    }

    public String getFeatureName() {
        return featureName;
    }

    public double getWeight() {
        return weight;
    }

    public double getFeatureValue() {
        return featureValue;
    }

    public double getScore() {
        return score;
    }
}
