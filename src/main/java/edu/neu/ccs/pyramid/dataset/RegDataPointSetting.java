package edu.neu.ccs.pyramid.dataset;

import java.io.Serializable;

/**
 * Created by chengli on 11/1/14.
 */
@Deprecated
public class RegDataPointSetting implements Serializable {
    private static final long serialVersionUID = 1L;
    private String extId = "unKnown";

    public RegDataPointSetting() {
    }

    public String getExtId() {
        return extId;
    }

    public RegDataPointSetting setExtId(String extId) {
        this.extId = extId;
        return this;
    }


    public RegDataPointSetting copy(){
        RegDataPointSetting dataSetting = new RegDataPointSetting();
        dataSetting.setExtId(this.extId);
        return dataSetting;
    }

    @Override
    public String toString() {
        return "RegDataPointSetting{" +
                "extId='" + extId + '\'' +
                '}';
    }
}
