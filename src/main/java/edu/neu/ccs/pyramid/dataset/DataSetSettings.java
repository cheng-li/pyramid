package edu.neu.ccs.pyramid.dataset;

import java.io.Serializable;
import java.util.List;

/**
 * Created by chengli on 3/5/15.
 */
@Deprecated
public class DataSetSettings implements Serializable {
    private static final long serialVersionUID = 1L;

    private IdTranslator idTranslator;
    private List<Feature> features;

    public IdTranslator getIdTranslator() {
        return idTranslator;
    }

    public void setIdTranslator(IdTranslator idTranslator) {
        this.idTranslator = idTranslator;
    }

    public List<Feature> getFeatures() {
        return features;
    }

    public void setFeatures(List<Feature> features) {
        this.features = features;
    }
}
