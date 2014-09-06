package edu.neu.ccs.pyramid.dataset;

import org.apache.mahout.math.RandomAccessSparseVector;

import org.apache.mahout.math.Vector;

/**
 * Created by chengli on 8/20/14.
 */
class SparseFeatureRow implements FeatureRow{
    int dataPointIndex;
    RandomAccessSparseVector vector;
    DataSetting setting;

    SparseFeatureRow(int dataPointIndex, int numFeatures) {
        this.dataPointIndex = dataPointIndex;
        this.vector = new RandomAccessSparseVector(numFeatures);
        this.setting = new DataSetting();
    }

    @Override
    public int getDataPointIndex() {
        return dataPointIndex;
    }

    @Override
    public Vector getVector() {
        return vector;
    }

    @Override
    public DataSetting getSetting() {
        return setting;
    }

    @Override
    public void putSetting(DataSetting setting) {
        this.setting = setting;
    }
}
