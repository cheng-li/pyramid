package edu.neu.ccs.pyramid.dataset;

import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.Vector;

/**
 * Created by chengli on 8/20/14.
 */
class DenseFeatureColumn implements FeatureColumn{
    int featureIndex;
    DenseVector vector;
    FeatureSetting setting;

    DenseFeatureColumn(int featureIndex, int numDataPoints) {
        this.featureIndex = featureIndex;
        this.vector = new DenseVector(numDataPoints);
        this.setting = new FeatureSetting();
    }

    @Override
    public int getFeatureIndex() {
        return featureIndex;
    }

    @Override
    public Vector getVector() {
        return vector;
    }

    @Override
    public FeatureSetting getSetting() {
        return setting;
    }
}
