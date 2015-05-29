package edu.neu.ccs.pyramid.regression.probabilistic_regression_tree;

import edu.neu.ccs.pyramid.feature.FeatureList;
import edu.neu.ccs.pyramid.regression.Regressor;
import org.apache.mahout.math.Vector;

/**
 * Created by chengli on 5/21/15.
 */
public class ProbRegStump implements Regressor{
    FeatureList featureList;
    GatingFunction gatingFunction;
    double leftOutput;
    double rightOutput;

    @Override
    public double predict(Vector vector) {
        double leftProb = gatingFunction.leftProbability(vector);
        return leftProb*leftOutput + (1-leftProb)*rightOutput;
    }

    @Override
    public FeatureList getFeatureList() {
        return featureList;
    }

    public GatingFunction getGatingFunction() {
        return gatingFunction;
    }

    public double getLeftOutput() {
        return leftOutput;
    }

    public double getRightOutput() {
        return rightOutput;
    }

    @Override
    public String toString() {
        final StringBuilder sb = new StringBuilder("ProbRegStump{");
        sb.append("gatingFunction=").append(gatingFunction);
        sb.append(", leftOutput=").append(leftOutput);
        sb.append(", rightOutput=").append(rightOutput);
        sb.append('}');
        return sb.toString();
    }

    public void shrink(double learningRate){
        leftOutput *= learningRate;
        rightOutput *= learningRate;
    }
}
