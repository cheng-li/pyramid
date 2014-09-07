package edu.neu.ccs.pyramid.feature;

/**
 * Created by chengli on 9/7/14.
 */
public class NumericalFeatureMapperBuilder {
    private String featureName = "non name";
    private int featureIndex;
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
