package edu.neu.ccs.pyramid.regression.probabilistic_regression_tree;

import org.apache.mahout.math.Vector;

/**
 * Created by chengli on 5/21/15.
 */
public interface GatingFunction {
    double leftProbability(Vector row);
    default double logLeftProbability(Vector row){
        return Math.log(logLeftProbability(row));
    }
    default double logRightProbability(Vector row){
        return Math.log(1-logLeftProbability(row));
    }
}
