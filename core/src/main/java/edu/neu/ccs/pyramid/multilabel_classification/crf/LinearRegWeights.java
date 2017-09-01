package edu.neu.ccs.pyramid.multilabel_classification.crf;

import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.SequentialAccessSparseVector;
import org.apache.mahout.math.Vector;

import java.io.*;

/**
 * Created by Rainicy on 11/12/16.
 */
public class LinearRegWeights {
    private int numFeatures;
    /**
     * vector is not serializable
     */
    private transient Vector weightVector;

    public LinearRegWeights(int numFeatures) {
        this.numFeatures = numFeatures;
        this.weightVector = new SequentialAccessSparseVector(numFeatures);
    }

    public LinearRegWeights(int numFeatures, Vector weightVector) {
        this.numFeatures = numFeatures;
        if (weightVector.size()!=numFeatures){
            throw new IllegalArgumentException("weightVector.size()!=(numFeatures)");
        }
        this.weightVector = weightVector;
    }

    /**
     *
     * @return weights including bias at the beginning
     */
    public Vector getWeights(){
        return this.weightVector;
    }

    public void setWeightVector(Vector weightVector) {
        this.weightVector = weightVector;
    }

    public void setWeight(int featureIndex, double weight){
        this.weightVector.set(featureIndex,weight);
    }

    @Override
    public String toString() {
        return "Weights{" +
                ", numFeatures=" + numFeatures +
                ", weightVector=" + weightVector +
                '}';
    }
}
