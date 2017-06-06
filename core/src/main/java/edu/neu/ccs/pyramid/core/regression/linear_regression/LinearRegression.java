package edu.neu.ccs.pyramid.core.regression.linear_regression;

import edu.neu.ccs.pyramid.core.feature.FeatureList;
import edu.neu.ccs.pyramid.core.regression.Regressor;
import org.apache.mahout.math.Vector;

import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;

/**
 * Created by chengli on 2/18/15.
 */
public class LinearRegression implements Regressor {
    private static final long serialVersionUID = 1L;

    private Weights weights;

    private FeatureList featureList;

    public LinearRegression(int numFeatures) {
        this.weights = new Weights(numFeatures);
    }

    /**
     *
     * @param numFeatures
     * @param weightVector the first element is the bias
     */
    public LinearRegression(int numFeatures, Vector weightVector){
        this.weights = new Weights(numFeatures,weightVector);
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

    public LinearRegression deepCopy() throws Exception{
        ByteArrayOutputStream baos = new ByteArrayOutputStream();
        ObjectOutputStream oos = new ObjectOutputStream(baos);
        oos.writeObject(this);
        oos.flush();
        ByteArrayInputStream bais = new ByteArrayInputStream(baos.toByteArray());
        ObjectInputStream ois = new ObjectInputStream(bais);
        LinearRegression linearRegression = (LinearRegression) ois.readObject();
        oos.close();
        ois.close();
        return linearRegression;
    }

    @Override
    public FeatureList getFeatureList() {
        return featureList;
    }

    void setFeatureList(FeatureList featureList) {
        this.featureList = featureList;
    }
}
