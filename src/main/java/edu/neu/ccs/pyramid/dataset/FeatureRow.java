package edu.neu.ccs.pyramid.dataset;

import org.apache.mahout.math.Vector;

/**
 * Created by chengli on 8/4/14.
 */
public interface FeatureRow {
    int getDataPointIndex();
    Vector getVector();
    Setting getSetting();
}
