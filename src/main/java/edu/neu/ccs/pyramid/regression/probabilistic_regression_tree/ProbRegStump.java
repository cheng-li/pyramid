package edu.neu.ccs.pyramid.regression.probabilistic_regression_tree;

import edu.neu.ccs.pyramid.feature.FeatureList;
import edu.neu.ccs.pyramid.regression.Regressor;
import org.apache.mahout.math.Vector;

/**
 * Created by chengli on 5/21/15.
 */
public class ProbRegStump implements Regressor{
    private FeatureList featureList;
    private GatingFunction gatingFunction;
    private double leftOutput;
    private double rightOutput;

    @Override
    public double predict(Vector vector) {
        double leftProb = gatingFunction.leftProbability(vector);
        return leftProb*leftOutput + (1-leftProb)*rightOutput;
    }

    @Override
    public FeatureList getFeatureList() {
        return null;
    }
}
