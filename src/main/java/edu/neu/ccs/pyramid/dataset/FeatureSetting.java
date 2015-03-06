package edu.neu.ccs.pyramid.dataset;

import java.io.Serializable;

/**
 * Created by chengli on 8/6/14.
 */
@Deprecated
public class FeatureSetting implements Serializable{
    private static final long serialVersionUID = 1L;

    private String featureName = "unKnown";
    private FeatureType featureType = FeatureType.NUMERICAL;

    public FeatureSetting() {
    }


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

    @Override
    public String toString() {
        return "FeatureSetting{" +
                "featureName='" + featureName + '\'' +
                ", featureType=" + featureType +
                '}';
    }

    public FeatureSetting copy(){
        FeatureSetting featureSetting = new FeatureSetting();
        featureSetting.setFeatureName(this.featureName);
        featureSetting.setFeatureType(this.featureType);
        return featureSetting;
    }
}
