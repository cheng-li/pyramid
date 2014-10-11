package edu.neu.ccs.pyramid.dataset;

import edu.neu.ccs.pyramid.feature.FeatureMappers;

import java.io.Serializable;
import java.util.HashMap;
import java.util.Map;

/**
 * only hold global data structures
 * Created by chengli on 9/6/14.
 */
public class DataSetSetting implements Serializable{
    private static final long serialVersionUID = 2L;

    private LabelTranslator labelTranslator;
    private IdTranslator idTranslator;
    private FeatureMappers featureMappers;


    public IdTranslator getIdTranslator() {
        return idTranslator;
    }

    public FeatureMappers getFeatureMappers() {
        return featureMappers;
    }

    public LabelTranslator getLabelTranslator() {
        return labelTranslator;
    }

    /**
     * users should use the utility method
     * @param labelTranslator
     */
    void setLabelTranslator(LabelTranslator labelTranslator) {
        this.labelTranslator = labelTranslator;
    }


    /**
     * users should use the utility method instead
     * @param idTranslator
     */
    void setIdTranslator(IdTranslator idTranslator) {
        this.idTranslator = idTranslator;
    }

    /**
     * users should use the utility method instead
     * @param featureMappers
     */
    void setFeatureMappers(FeatureMappers featureMappers) {
        this.featureMappers = featureMappers;
    }

    public DataSetSetting copy(){
        DataSetSetting dataSetSetting = new DataSetSetting();
        dataSetSetting.setFeatureMappers(this.featureMappers);
        dataSetSetting.setIdTranslator(this.idTranslator);
        dataSetSetting.setLabelTranslator(this.labelTranslator);
        return dataSetSetting;
    }
}
