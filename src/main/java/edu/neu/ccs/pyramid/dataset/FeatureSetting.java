package edu.neu.ccs.pyramid.dataset;

import edu.neu.ccs.pyramid.feature.FeatureType;

/**
 * Created by chengli on 8/6/14.
 */
public class FeatureSetting {
    private FeatureType featureType = FeatureType.NUMERICAL;

    public FeatureType getFeatureType() {
        return featureType;
    }

    public FeatureSetting setFeatureType(FeatureType featureType) {
        this.featureType = featureType;
        return this;
    }
}
