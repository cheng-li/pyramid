package edu.neu.ccs.pyramid.feature;

import java.io.Serializable;

/**
 * Created by chengli on 9/7/14.
 */
public class NumericalFeatureMapper implements Serializable {
    private static final long serialVersionUID = 1L;
    private String featureName;
    private int featureIndex;
    private String source;

    protected NumericalFeatureMapper(String featureName, int featureIndex, String source) {
        this.featureName = featureName;
        this.featureIndex = featureIndex;
        this.source = source;
    }

    public String getFeatureName() {
        return featureName;
    }

    public int getFeatureIndex() {
        return featureIndex;
    }

    public String getSource() {
        return source;
    }

    public static NumericalFeatureMapperBuilder getBuilder(){
        return new NumericalFeatureMapperBuilder();
    }

}
