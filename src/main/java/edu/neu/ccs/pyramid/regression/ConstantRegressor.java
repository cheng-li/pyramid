package edu.neu.ccs.pyramid.regression;


import org.apache.mahout.math.Vector;

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

    public double getScore() {
        return score;
    }

    @Override
    public double predict(Vector vector) {
        return this.score;
    }

    @Override
    public String toString() {
        return "ConstantRegressor{" +
                "score=" + score +
                '}' +"\n";
    }
}
