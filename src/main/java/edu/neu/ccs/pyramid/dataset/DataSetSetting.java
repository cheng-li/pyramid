package edu.neu.ccs.pyramid.dataset;

import edu.neu.ccs.pyramid.elasticsearch.IdTranslator;

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

    public Map<Integer, String> getLabelMap() {
        return labelMap;
    }

    public void setLabelMap(Map<Integer, String> labelMap) {
        this.labelMap = labelMap;
    }

    public void setLabelMap(String[] extLabels){
        this.labelMap = new HashMap<>();
        for (int i=0;i<extLabels.length;i++){
            this.labelMap.put(i,extLabels[i]);
        }
    }


    public IdTranslator getIdTranslator() {
        return idTranslator;
    }

    /**
     * should use the utility method instead
     * @param idTranslator
     */
    void setIdTranslator(IdTranslator idTranslator) {
        this.idTranslator = idTranslator;
    }
}
