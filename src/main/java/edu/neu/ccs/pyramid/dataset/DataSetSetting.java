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
    private static final long serialVersionUID = 1L;

    private Map<Integer, String> labelMap;
    private IdTranslator idTranslator;
    private FeatureMappers featureMappers;

    public Map<Integer, String> getLabelMap() {
        return labelMap;
    }

    public IdTranslator getIdTranslator() {
        return idTranslator;
    }

    public FeatureMappers getFeatureMappers() {
        return featureMappers;
    }

    /**
     * just for ClfDataSet
     * users should use the utility method
     * @param labelMap
     */
    void setLabelMap(Map<Integer, String> labelMap) {
        this.labelMap = labelMap;
    }

    /***
     * just for ClfDataSet
     * users should use the utility method
     * @param extLabels
     */
    void setLabelMap(String[] extLabels){
        this.labelMap = new HashMap<>();
        for (int i=0;i<extLabels.length;i++){
            this.labelMap.put(i,extLabels[i]);
        }
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
        dataSetSetting.setLabelMap(this.labelMap);
        return dataSetSetting;
    }
}
