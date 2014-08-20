package edu.neu.ccs.pyramid.dataset;

import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.Vector;

/**
 * Created by chengli on 8/20/14.
 */
class DenseFeatureRow implements FeatureRow{
    int dataPointIndex;
    DenseVector vector;
    DataSetting setting;

    DenseFeatureRow(int dataPointIndex, int numFeatures) {
        this.dataPointIndex = dataPointIndex;
        this.vector = new DenseVector(numFeatures);
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
}
