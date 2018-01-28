package edu.neu.ccs.pyramid.feature;

import edu.neu.ccs.pyramid.feature_selection.FeatureDistribution;

import java.util.List;

/**
 * Created by chengli on 3/14/15.
 */
public class TopFeatures {
    private int classIndex;
    private String className;
    private List<Feature> topFeatures;
    private List<Double> utilities;
//    private List<FeatureDistribution> featureDistributions;

    public TopFeatures() {
    }

    public List<Double> getUtilities() {
        return utilities;
    }

    public void setUtilities(List<Double> utilities) {
        this.utilities = utilities;
    }

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

    public List<Feature> getTopFeatures() {
        return topFeatures;
    }

    public void setTopFeatures(List<Feature> topFeatures) {
        this.topFeatures = topFeatures;
    }

//    public List<FeatureDistribution> getFeatureDistributions() {
//        return featureDistributions;
//    }
//
//    public void setFeatureDistributions(List<FeatureDistribution> featureDistributions) {
//        this.featureDistributions = featureDistributions;
//    }
}
