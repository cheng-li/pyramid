package edu.neu.ccs.pyramid.dataset;

import org.apache.mahout.math.RandomAccessSparseVector;
import org.apache.mahout.math.Vector;

/**
 * Created by chengli on 8/4/14.
 */
class SparseFeatureColumn implements FeatureColumn{
    int featureIndex;
    RandomAccessSparseVector vector;
    FeatureSetting setting;

    SparseFeatureColumn(int featureIndex, int numDataPoints) {
        this.featureIndex = featureIndex;
        this.vector = new RandomAccessSparseVector(numDataPoints);
        this.setting = new FeatureSetting();
    }

    @Override
    public int getFeatureIndex() {
        return this.featureIndex;
    }

    @Override
    public Vector getVector() {
        return this.vector;
    }

    @Override
    public FeatureSetting getSetting() {
        return this.setting;
    }

    @Override
    public void putSetting(FeatureSetting setting) {
        this.setting = setting;
    }
}
