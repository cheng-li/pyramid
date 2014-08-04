package edu.neu.ccs.pyramid.dataset;

import org.apache.mahout.math.Vector;

/**
 * Created by chengli on 8/4/14.
 */
public interface FeatureColumn {
    int getFeatureIndex();
    Vector getVector();
    Setting getSetting();

}
