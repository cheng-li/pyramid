package edu.neu.ccs.pyramid.dataset;

import org.apache.mahout.math.Vector;

/**
 * Created by chengli on 8/4/14.
 */
public interface FeatureRow {
    int getDataPointIndex();
    Vector getVector();
    DataSetting getSetting();
    default String print(){
        StringBuilder sb = new StringBuilder();
        sb.append("data point index = ").append(getDataPointIndex()).append("\n");
        sb.append("vector = ").append(getVector()).append("\n");
        sb.append("setting = ").append(getSetting());
        return sb.toString();
    }

}
