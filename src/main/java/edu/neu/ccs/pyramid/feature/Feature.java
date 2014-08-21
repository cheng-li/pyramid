package edu.neu.ccs.pyramid.feature;

/**
 * Created by chengli on 8/20/14.
 */
public class Feature {
    protected String featureName;

    public Feature(String featureName) {
        this.featureName = featureName;
    }


    public String getFeatureName() {
        return featureName;
    }

    public void setFeatureName(String featureName) {
        this.featureName = featureName;
    }

    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder("Feature{");
        sb.append(", featureName='").append(featureName).append('\'');
        sb.append('}');
        return sb.toString();
    }

}
