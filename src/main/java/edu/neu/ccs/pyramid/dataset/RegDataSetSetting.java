package edu.neu.ccs.pyramid.dataset;

import edu.neu.ccs.pyramid.feature.FeatureMappers;

import java.io.Serializable;

/**
 * Created by chengli on 11/1/14.
 */
@Deprecated
public class RegDataSetSetting implements Serializable {
    private static final long serialVersionUID = 1L;

    private IdTranslator idTranslator;
    private FeatureMappers featureMappers;


    public IdTranslator getIdTranslator() {
        return idTranslator;
    }

    public FeatureMappers getFeatureMappers() {
        return featureMappers;
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

    public RegDataSetSetting copy(){
        RegDataSetSetting dataSetSetting = new RegDataSetSetting();
        dataSetSetting.setFeatureMappers(this.featureMappers);
        dataSetSetting.setIdTranslator(this.idTranslator);
        return dataSetSetting;
    }
}
