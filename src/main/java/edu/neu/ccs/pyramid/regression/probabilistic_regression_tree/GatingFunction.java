package edu.neu.ccs.pyramid.regression.probabilistic_regression_tree;

import org.apache.mahout.math.Vector;

/**
 * Created by chengli on 5/21/15.
 */
public interface GatingFunction {
    double leftProbability(Vector row);
}
