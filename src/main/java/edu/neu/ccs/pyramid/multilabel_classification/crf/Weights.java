package edu.neu.ccs.pyramid.multilabel_classification.crf;

import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.VectorView;

import java.io.Serializable;

/**
 * Created by Rainicy on 12/12/15.
 */
public class Weights  implements Serializable {
    private static final long serialVersionUID = 2L;
    private int numFeatures;
    private int numClasses;

    private int numWeightsForFeatures;
    private int numWeightsForLabels;
    /**
     * size = (numFeatures + 1) * numClasses +
     * (numClasses choose 2) * 4 *
     * where equals number of weights features and plus
     * the pair-wise labels, which has 4 possible combinations
     * vector is not serializable
     */
    private transient Vector weightVector;
    /**
     * serialize this array instead
     */
    private double[] serializableWeights;

    public Weights(int numClasses, int numFeatures) {
        this.numClasses = numClasses;
        this.numFeatures = numFeatures;
        this.numWeightsForFeatures = (numFeatures + 1) * numClasses;
        this.numWeightsForLabels = (numClasses * (numClasses-1)/2) * 4;
        this.weightVector = new DenseVector(numWeightsForFeatures + numWeightsForLabels);
        this.serializableWeights = new double[numWeightsForFeatures + numWeightsForLabels];
    }

    public Weights deepCopy(){
        Weights copy = new Weights(this.numClasses,numFeatures);
        copy.weightVector = new DenseVector(this.weightVector);
        return copy;
    }

    /**
     * @param parameterIndex
     * @return the class index
     */
    public int getClassIndex(int parameterIndex){
        return parameterIndex/(numFeatures+1);
    }

    /**
     * @param parameterIndex
     * @return feature index
     */
    public int getFeatureIndex(int parameterIndex){
        return parameterIndex - getClassIndex(parameterIndex)*(numFeatures+1) -1;
    }


    public int totalSize(){
        return weightVector.size();
    }


    public void setWeightVector(Vector weightVector) {
        if (weightVector.size() != (numWeightsForFeatures + numWeightsForLabels)) {
            throw new IllegalArgumentException("given vector size is wrong: " + weightVector.size());
        }
        this.weightVector = weightVector;
    }

    /**
     * @return weights for all classes
     */
    public Vector getAllWeights() {
        return weightVector;
    }

    public Vector getAllLabelPairWeights() {
        return new VectorView(this.weightVector, numWeightsForFeatures, numWeightsForLabels);
    }


    /**
     * return the weights by given class k.
     * @param k
     * @return
     */
    public Vector getWeightsForClass(int k){
        int start = (this.numFeatures+1)*k;
        int length = this.numFeatures + 1;
        return new VectorView(this.weightVector,start,length);
    }

    /**
     * get the weights without bias.
     * @param k
     * @return weights for class k, no bias
     */
    public Vector getWeightsWithoutBiasForClass(int k){
        int start = (this.numFeatures+1)*k + 1;
        int length = this.numFeatures;
        return new VectorView(this.weightVector,start,length);
    }

    /**
     * get bias for given class k.
     * @param k
     * @return
     */
    public double getBiasForClass(int k){
        int start = (this.numFeatures+1)*k;
        return this.weightVector.get(start);
    }

    public double getWeightForIndex(int index) {
        return this.weightVector.get(index);
    }

    public int getNumWeightsForFeatures() {
        return numWeightsForFeatures;
    }

    public int getNumWeightsForLabels() {
        return numWeightsForLabels;
    }

    @Override
    public String toString() {
        final StringBuilder sb = new StringBuilder("Weights{");
        for (int k=0;k<numClasses;k++){
            sb.append("for class ").append(k).append(":").append("\n");
            sb.append("bias = "+getBiasForClass(k)).append(",");
            sb.append("weights = "+getWeightsWithoutBiasForClass(k)).append("\n");
        }
        sb.append('}');
        return sb.toString();
    }
}
