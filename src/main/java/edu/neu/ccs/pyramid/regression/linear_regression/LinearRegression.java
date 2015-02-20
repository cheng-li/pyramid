package edu.neu.ccs.pyramid.regression.linear_regression;

import edu.neu.ccs.pyramid.regression.Regressor;
import org.apache.mahout.math.Vector;

/**
 * Created by chengli on 2/18/15.
 */
public class LinearRegression implements Regressor{
    private Weights weights;

    public LinearRegression(int numFeatures) {
        this.weights = new Weights(numFeatures);
    }

    public Weights getWeights() {
        return weights;
    }

    @Override
    public double predict(Vector vector) {
        double score = 0;
        score += this.weights.getBias();
        score += this.weights.getWeightsWithoutBias().dot(vector);
        return score;
    }

    public double predictWithoutBias(Vector vector){
        return this.weights.getWeightsWithoutBias().dot(vector);
    }

}
