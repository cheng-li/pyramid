package edu.neu.ccs.pyramid.regression;


import edu.neu.ccs.pyramid.dataset.LabelTranslator;
import edu.neu.ccs.pyramid.feature.FeatureList;
import org.apache.mahout.math.Vector;

import java.io.Serializable;

/**
 * Created by chengli on 8/19/14.
 */
public class ConstantRegressor implements Regressor, Serializable{
    private static final long serialVersionUID = 1L;

    private double score;
    private FeatureList featureList;

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

    @Override
    public FeatureList getFeatureList() {
        return featureList;
    }

    public void setFeatureList(FeatureList featureList) {
        this.featureList = featureList;
    }
}
