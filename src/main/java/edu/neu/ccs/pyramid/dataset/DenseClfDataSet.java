package edu.neu.ccs.pyramid.dataset;

import java.util.Arrays;

/**
 * Created by chengli on 8/14/14.
 */
public class DenseClfDataSet extends DenseDataSet implements ClfDataSet{
    int numClasses;
    private int[] labels;
    private ClfDataSetSetting dataSetSetting;
    private ClfDataPointSetting[] dataPointSettings;

    DenseClfDataSet(int numDataPoints, int numFeatures,
                           boolean missingValue, int numClasses) {
        super(numDataPoints, numFeatures, missingValue);
        this.labels = new int[numDataPoints];
        this.numClasses = numClasses;
        this.dataSetSetting = new ClfDataSetSetting();
        this.dataPointSettings = new ClfDataPointSetting[numDataPoints];
        for (int i=0;i<numDataPoints;i++){
            this.dataPointSettings[i] = new ClfDataPointSetting();
        }
    }

    @Override
    public int getNumClasses() {
        return this.numClasses;
    }

    @Override
    public int[] getLabels() {
        return this.labels;
    }

    @Override
    public void setLabel(int dataPointIndex, int label) {
        if (label<0||label>=this.numClasses){
            throw new IllegalArgumentException("label<0||label>=this.numClasses");
        }
        this.labels[dataPointIndex]=label;
    }

    @Override
    public ClfDataSetSetting getSetting() {
        return this.dataSetSetting;
    }

    @Override
    public ClfDataPointSetting getDataPointSetting(int dataPointIndex) {
        return this.dataPointSettings[dataPointIndex];
    }

    @Override
    public void putDataSetSetting(ClfDataSetSetting dataSetSetting) {
        this.dataSetSetting = dataSetSetting;
    }

    @Override
    public void putDataPointSetting(int dataPointIndex, ClfDataPointSetting dataPointSetting) {
        this.dataPointSettings[dataPointIndex] = dataPointSetting;
    }

    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder();
        sb.append("number of classes = ").append(this.numClasses).append("\n");
        sb.append(super.toString());
        sb.append("labels = ").append(Arrays.toString(labels));
        return sb.toString();
    }

    @Override
    public String getMetaInfo() {
        StringBuilder sb = new StringBuilder();
        sb.append(super.getMetaInfo());
        sb.append("type = ").append("dense classification").append("\n");
        sb.append("number of classes = ").append(this.numClasses);
        return sb.toString();
    }
}
