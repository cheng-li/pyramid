package edu.neu.ccs.pyramid.regression;


import org.apache.mahout.math.Vector;

/**
 * Created by chengli on 8/6/14.
 */
public interface Regressor {
    double predict(Vector vector);
}
