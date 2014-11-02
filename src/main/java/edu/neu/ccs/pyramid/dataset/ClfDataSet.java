package edu.neu.ccs.pyramid.dataset;

/**
 * Created by chengli on 8/13/14.
 */
public interface ClfDataSet extends DataSet{
    int getNumClasses();
    int[] getLabels();
    void setLabel(int dataPointIndex, int label);
    ClfDataSetSetting getSetting();
    ClfDataPointSetting getDataPointSetting(int dataPointIndex);
    void putDataSetSetting(ClfDataSetSetting dataSetSetting);
    void putDataPointSetting(int dataPointIndex, ClfDataPointSetting dataPointSetting);
}
