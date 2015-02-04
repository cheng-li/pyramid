package edu.neu.ccs.pyramid.feature;

import java.util.Map;

/**
 * Created by chengli on 9/7/14.
 */
public class NumericalFeatureMapperBuilder {
    private String featureName = "no name";
    private int featureIndex;
    @Deprecated
    private String source = "unknown";
    private boolean featureIndexSet = false;

    public NumericalFeatureMapperBuilder() {
    }

    public NumericalFeatureMapperBuilder setFeatureName(String featureName) {
        this.featureName = featureName;
        return this;
    }

    public NumericalFeatureMapperBuilder setFeatureIndex(int featureIndex) {
        this.featureIndex = featureIndex;
        this.featureIndexSet = true;
        return this;
    }

    @Deprecated
    public NumericalFeatureMapperBuilder setSource(String source) {
        this.source = source;
        return this;
    }

    public NumericalFeatureMapper build(){
        if (!this.featureIndexSet){
            throw new RuntimeException("feature index not set yet!");
        }
        return new NumericalFeatureMapper(this.featureName,this.featureIndex,this.source);
    }


}
