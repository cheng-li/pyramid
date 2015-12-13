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
    /**
     * size = numFeatures * 2 * numClasses;
     * each label has 2 * numClasses: binary classification.
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
        this.weightVector = new DenseVector(numFeatures * numClasses * 2);
        this.serializableWeights = new double[numFeatures * numClasses * 2];
    }

    public Weights deepCopy(){
        Weights copy = new Weights(this.numClasses,numFeatures);
        copy.weightVector = new DenseVector(this.weightVector);
        return copy;
    }

    /**
     * TODO: Maybe wrong.
     * @param parameterIndex
     * @return the class index
     */
    public int getClassIndex(int parameterIndex){
        return parameterIndex/numFeatures/2;
    }

    /**
     * TODO: maybe wrong.
     * @param parameterIndex
     * @return feature index
     */
    public int getFeatureIndex(int parameterIndex){
        int doubleFeatureIndex = parameterIndex - getClassIndex(parameterIndex)*numFeatures*2;
        if (doubleFeatureIndex < numFeatures) {
            return doubleFeatureIndex;
        } else {
            return (doubleFeatureIndex - numFeatures);
        }
    }


    public void setWeightVector(Vector weightVector) {
        if (weightVector.size() != (numFeatures * numClasses * 2)) {
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


    /**
     * return the weights of binary labels by given class k.
     * @param k
     * @return
     */
    public Vector getWeightsForClass(int k, int l){
        int start = this.numFeatures * k * 2 + l * this.numFeatures;
        int length = this.numFeatures;
        return new VectorView(this.weightVector,start,length);
    }


    @Override
    public String toString() {
        final StringBuilder sb = new StringBuilder("Weights{");
        for (int k=0;k<numClasses;k++){
            sb.append("for label ").append(k).append(":").append("\n");
            sb.append("negative weights = "+getWeightsForClass(k, 0)).append("\t");
            sb.append("positive weights = "+getWeightsForClass(k, 1)).append("\n");
        }
        sb.append('}');
        return sb.toString();
    }
}
