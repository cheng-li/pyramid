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
    DataSetSetting getSetting();
    void putSetting(DataSetSetting setting);
    default String getMetaInfo(){
        StringBuilder sb = new StringBuilder();
        sb.append("data set meta information:").append("\n");
        sb.append("number of data points = ").append(getNumDataPoints()).append("\n");
        sb.append("number of features = ").append(getNumFeatures()).append("\n");
        return sb.toString();
    }

}
