package edu.neu.ccs.pyramid.dataset;

/**
 * Created by chengli on 8/7/14.
 */
public interface RegDataSet extends DataSet{
    public double[] getLabels();
    public void setLabel(int dataPointIndex, double label);
    RegDataSetSetting getSetting();
    RegDataPointSetting getDataPointSetting(int dataPointIndex);
    void putDataSetSetting(RegDataSetSetting dataSetSetting);
    void putDataPointSetting(int dataPointIndex, RegDataPointSetting dataPointSetting);
}
