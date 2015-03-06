package edu.neu.ccs.pyramid.dataset;

import java.io.Serializable;


/**
 * Created by chengli on 11/1/14.
 */
@Deprecated
public class ClfDataPointSetting implements Serializable {
    private static final long serialVersionUID = 1L;
    private String extId = "unKnown";
    private String extLabel = "unKnown";

    public ClfDataPointSetting() {
    }

    public String getExtId() {
        return extId;
    }

    public ClfDataPointSetting setExtId(String extId) {
        this.extId = extId;
        return this;
    }

    public String getExtLabel() {
        return extLabel;
    }

    public ClfDataPointSetting setExtLabel(String extLabel) {
        this.extLabel = extLabel;
        return this;
    }



    public ClfDataPointSetting copy(){
        ClfDataPointSetting dataSetting = new ClfDataPointSetting();
        dataSetting.setExtId(this.extId);
        dataSetting.setExtLabel(this.extLabel);
        return dataSetting;
    }

    @Override
    public String toString() {
        return "ClfDataPointSetting{" +
                "extId='" + extId + '\'' +
                ", extLabel='" + extLabel + '\'' +
                '}';
    }
}
