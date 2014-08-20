package edu.neu.ccs.pyramid.regression;

import edu.neu.ccs.pyramid.dataset.FeatureRow;

import java.io.Serializable;

/**
 * Created by chengli on 8/19/14.
 */
public class ConstantRegressor implements Regressor, Serializable{
    private static final long serialVersionUID = 1L;

    private double score;

    public ConstantRegressor(double score) {
        this.score = score;
    }

    @Override
    public double predict(FeatureRow featureRow) {
        return this.score;
    }

    @Override
    public String toString() {
        return "ConstantRegressor{" +
                "score=" + score +
                '}';
    }
}
