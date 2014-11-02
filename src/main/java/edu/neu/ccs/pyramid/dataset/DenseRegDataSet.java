package edu.neu.ccs.pyramid.dataset;

import java.util.Arrays;

/**
 * Created by chengli on 8/7/14.
 */
public class DenseRegDataSet extends DenseDataSet implements RegDataSet {
    private double[] labels;
    private RegDataSetSetting dataSetSetting;
    private RegDataPointSetting[] dataPointSettings;

    DenseRegDataSet(int numDataPoints, int numFeatures, boolean missingValue) {
        super(numDataPoints, numFeatures, missingValue);
        this.labels = new double[numDataPoints];
        this.dataSetSetting = new RegDataSetSetting();
        this.dataPointSettings = new RegDataPointSetting[numDataPoints];
        for (int i=0;i<numDataPoints;i++){
            this.dataPointSettings[i] = new RegDataPointSetting();
        }
    }

    @Override
    public double[] getLabels() {
        return this.labels;
    }

    @Override
    public void setLabel(int dataPointIndex, double label) {
        this.labels[dataPointIndex]=label;
    }

    @Override
    public RegDataSetSetting getSetting() {
        return this.dataSetSetting;
    }

    @Override
    public RegDataPointSetting getDataPointSetting(int dataPointIndex) {
        return this.dataPointSettings[dataPointIndex];
    }

    @Override
    public void putDataSetSetting(RegDataSetSetting dataSetSetting) {
        this.dataSetSetting = dataSetSetting;
    }

    @Override
    public void putDataPointSetting(int dataPointIndex, RegDataPointSetting dataPointSetting) {
        this.dataPointSettings[dataPointIndex] = dataPointSetting;
    }

    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder();
        sb.append(super.toString());
        sb.append("labels = ").append(Arrays.toString(labels));
        return sb.toString();
    }

    @Override
    public String getMetaInfo() {
        StringBuilder sb = new StringBuilder();
        sb.append(super.getMetaInfo());
        sb.append("type = ").append("dense regression");
        return sb.toString();
    }
}
