package edu.neu.ccs.pyramid.dataset;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.List;

/**
 * Created by chengli on 11/1/14.
 */
public class MLClfDataPointSetting implements Serializable{
    private static final long serialVersionUID = 1L;
    private String extId = "unKnown";
    private List<String> extLabels;

    public MLClfDataPointSetting() {
        this.extLabels = new ArrayList<>();
    }

    public String getExtId() {
        return extId;
    }

    public MLClfDataPointSetting setExtId(String extId) {
        this.extId = extId;
        return this;
    }


    public List<String> getExtLabels() {
        return extLabels;
    }

    public void setExtLabels(List<String> extLabels) {
        this.extLabels = extLabels;
    }

    public MLClfDataPointSetting copy(){
        MLClfDataPointSetting dataSetting = new MLClfDataPointSetting();
        dataSetting.setExtId(this.extId);
        for (int i=0;i<this.extLabels.size();i++){
            dataSetting.extLabels.add(this.extLabels.get(i));
        }
        return dataSetting;
    }

    @Override
    public String toString() {
        return "MLClfDataPointSetting{" +
                "extId='" + extId + '\'' +
                ", extLabels=" + extLabels +
                '}';
    }
}
