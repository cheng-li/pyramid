package edu.neu.ccs.pyramid.dataset;


/**
 * Created by chengli on 8/4/14.
 */
public interface DataSet {
    int getNumDataPoints();
    int getNumFeatures();
    FeatureColumn getFeatureColumn(int featureIndex);
    FeatureRow getFeatureRow(int dataPointIndex);
    void setFeatureValue(int dataPointIndex,
                                int featureIndex, double featureValue);
    boolean isDense();
    boolean hasMissingValue();
    DataSetSetting getSetting();
    void putSetting(DataSetSetting setting);
    String getMetaInfo();

}
