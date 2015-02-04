package edu.neu.ccs.pyramid.feature;

import java.io.Serializable;
import java.util.HashMap;
import java.util.Map;

/**
 * Created by chengli on 9/7/14.
 */
public class NumericalFeatureMapper implements Serializable {
    private static final long serialVersionUID = 2L;
    private String featureName;
    private int featureIndex;
    @Deprecated
    private String source;
    private Map<String,String> settings;

    protected NumericalFeatureMapper(String featureName, int featureIndex, String source) {
        this.featureName = featureName;
        this.featureIndex = featureIndex;
        this.source = source;
        this.settings = new HashMap<>();
    }

    public Map<String, String> getSettings() {
        return settings;
    }

    public String getFeatureName() {
        return featureName;
    }

    public int getFeatureIndex() {
        return featureIndex;
    }

    @Deprecated
    public String getSource() {
        return source;
    }

    public static NumericalFeatureMapperBuilder getBuilder(){
        return new NumericalFeatureMapperBuilder();
    }

}
