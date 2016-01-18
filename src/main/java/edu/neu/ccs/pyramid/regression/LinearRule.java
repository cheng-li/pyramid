package edu.neu.ccs.pyramid.regression;

import edu.neu.ccs.pyramid.feature.Feature;

/**
 * Created by chengli on 2/28/15.
 */
public class LinearRule implements Rule {
    private Feature feature;
    private double weight;
    private double featureValue;
    private double score;

    public LinearRule() {
    }

    public Feature getFeature() {
        return feature;
    }

    public void setFeature(Feature feature) {
        this.feature = feature;
    }

    public double getWeight() {
        return weight;
    }

    public void setWeight(double weight) {
        this.weight = weight;
    }

    public double getFeatureValue() {
        return featureValue;
    }

    public void setFeatureValue(double featureValue) {
        this.featureValue = featureValue;
    }

    public double getScore() {
        return score;
    }

    public void setScore(double score) {
        this.score = score;
    }

    @Override
    public String toString() {
        final StringBuilder sb = new StringBuilder("LinearRule{");
        sb.append("feature=").append(feature);
        sb.append(", weight=").append(weight);
        sb.append(", featureValue=").append(featureValue);
        sb.append(", score=").append(score);
        sb.append('}');
        return sb.toString();
    }
}
