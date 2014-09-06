package edu.neu.ccs.pyramid.dataset;

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
}
