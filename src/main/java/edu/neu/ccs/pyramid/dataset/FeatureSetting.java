package edu.neu.ccs.pyramid.dataset;

import edu.neu.ccs.pyramid.feature.FeatureType;

import java.io.Serializable;

/**
 * Created by chengli on 8/6/14.
 */
public class FeatureSetting implements Serializable{
    private static final long serialVersionUID = 1L;

    private FeatureType featureType = FeatureType.NUMERICAL;
    private String featureName = "unKnown";

    public FeatureType getFeatureType() {
        return featureType;
    }

    public FeatureSetting setFeatureType(FeatureType featureType) {
        this.featureType = featureType;
        return this;
    }

    public String getFeatureName() {
        return featureName;
    }

    public FeatureSetting setFeatureName(String featureName) {
        this.featureName = featureName;
        return this;
    }
}
