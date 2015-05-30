package edu.neu.ccs.pyramid.regression.probabilistic_regression_tree;

import org.apache.mahout.math.Vector;

/**
 * Created by chengli on 5/29/15.
 */
public class HardThreshold implements GatingFunction {
    private int featureIndex;
    private double threshold;

    public HardThreshold(int featureIndex, double threshold) {
        this.featureIndex = featureIndex;
        this.threshold = threshold;
    }

    @Override
    public double leftProbability(Vector row) {
        if (row.get(featureIndex)<=threshold){
            return 1;
        } else {
            return 0;
        }
    }
}
