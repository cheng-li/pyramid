package edu.neu.ccs.pyramid.regression;


import org.apache.mahout.math.Vector;

import java.io.Serializable;

/**
 * Created by chengli on 8/6/14.
 */
public interface Regressor extends Serializable {
    double predict(Vector vector);
}
