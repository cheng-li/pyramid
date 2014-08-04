package edu.neu.ccs.pyramid.dataset;

import org.apache.mahout.math.RandomAccessSparseVector;
import org.apache.mahout.math.Vector;

/**
 * Created by chengli on 8/4/14.
 */
public class SparseFeatureColumn implements FeatureColumn{
    private int featureIndex;
    private RandomAccessSparseVector vector;
    private Setting setting;
    @Override
    public int getFeatureIndex() {
        return this.featureIndex;
    }

    @Override
    public Vector getVector() {
        return this.vector;
    }

    @Override
    public Setting getSetting() {
        return this.setting;
    }
}
