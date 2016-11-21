package edu.neu.ccs.pyramid.multilabel_classification.crf;

import edu.neu.ccs.pyramid.feature.FeatureList;
import edu.neu.ccs.pyramid.regression.Regressor;
import org.apache.mahout.math.Vector;

/**
 * Created by Rainicy on 11/12/16.
 */
public class CRFLinearRegression implements Regressor {

    private LinearRegWeights weights;

    private FeatureList featureList;

    public CRFLinearRegression(int numFeatures) {
        this.weights = new LinearRegWeights(numFeatures);
    }

    public CRFLinearRegression(int numFeatures, Vector weightVector) {
        this.weights = new LinearRegWeights(numFeatures, weightVector);
    }

    public LinearRegWeights getWeights() {
        return weights;
    }

    @Override
    public double predict(Vector vector) {
        return this.weights.getWeights().dot(vector);
    }

    @Override
    public FeatureList getFeatureList() {
        return null;
    }
}
