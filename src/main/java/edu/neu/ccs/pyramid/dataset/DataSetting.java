package edu.neu.ccs.pyramid.dataset;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.List;

/**
 * Created by chengli on 8/6/14.
 */
public class DataSetting implements Serializable{
    private static final long serialVersionUID = 2L;
    private String extId = "unKnown";
    private String extLabel = "unKnown";
    private List<String> extLabels = new ArrayList<>();

    public DataSetting() {
    }

    public String getExtId() {
        return extId;
    }

    public DataSetting setExtId(String extId) {
        this.extId = extId;
        return this;
    }

    public String getExtLabel() {
        return extLabel;
    }

    public DataSetting setExtLabel(String extLabel) {
        this.extLabel = extLabel;
        return this;
    }

    public List<String> getExtLabels() {
        return extLabels;
    }

    public void setExtLabels(List<String> extLabels) {
        this.extLabels = extLabels;
    }




    public DataSetting copy(){
        DataSetting dataSetting = new DataSetting();
        dataSetting.setExtId(this.extId);
        dataSetting.setExtLabel(this.extLabel);
        return dataSetting;
    }

    @Override
    public String toString() {
        return "DataSetting{" +
                "extId='" + extId + '\'' +
                ", extLabel='" + extLabel + '\'' +
                ", extLabels=" + extLabels +
                '}';
    }
}
