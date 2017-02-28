package edu.neu.ccs.pyramid.multilabel_classification.cbm;

import edu.neu.ccs.pyramid.dataset.SerializableVector;
import edu.neu.ccs.pyramid.util.MathUtil;
import edu.neu.ccs.pyramid.util.Vectors;
import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.Vector;

import java.io.IOException;
import java.io.Serializable;

/**
 * binary logistic regression on augmented vector (x,z)
 * Created by chengli on 2/28/17.
 */
public class AugmentedLR implements Serializable{
    private static final long serialVersionUID = 1L;

    private int numFeatures;
    private int numComponents;
    // size = num features + num components + 1
    // assuming the last number is the global bias
    private transient Vector weights;

    public AugmentedLR(int numFeatures, int numComponents) {
        this.numFeatures = numFeatures;
        this.numComponents = numComponents;
        this.weights = new DenseVector(numFeatures + numComponents +1);
    }

    public int getNumComponents() {
        return numComponents;
    }

    Vector getAllWeights() {
        return weights;
    }

    void setWeights(Vector weights) {
        this.weights = weights;
    }

    private double getWeightForComponent(int k){
        return weights.get(numFeatures+k);
    }

    private double getBias(){
        return weights.get(weights.size()-1);
    }

    Vector featureWeights(){
        return weights.viewPart(0, numFeatures);
    }

    Vector getWeightsWithoutBias(){
        return weights.viewPart(0, numFeatures+numComponents);
    }

    /**
     * feature part score, including the global bias
     * @param featureVector
     * @return
     */
    private double featureScore(Vector featureVector){
        return Vectors.dot(featureWeights(), featureVector) + getBias();
    }

    /**
     * total score for each (x,z)
     * @param featureVector
     * @return
     */
    private double[] augmentedScores(Vector featureVector){
        double[] scores = new double[numComponents];
        double featureScore = featureScore(featureVector);
        for (int k=0;k<numComponents;k++){
            scores[k] = featureScore + getWeightForComponent(k);
        }
        return scores;
    }

    private double[][] logAugmentedProbs(double[] augmentedScores){
        double[][] logProbs = new double[numComponents][2];
        for (int k=0;k<numComponents;k++){
            double[] s = {0, augmentedScores[k]};
            logProbs[k] = MathUtil.softmax(s);
        }
        return logProbs;
    }

    double[][] logAugmentedProbs(Vector featureVector){
        double[] scores = augmentedScores(featureVector);
        return logAugmentedProbs(scores);
    }



    private void writeObject(java.io.ObjectOutputStream out) throws IOException {
        out.writeInt(numFeatures);
        out.writeInt(numComponents);
        SerializableVector serializableVector = new SerializableVector(weights);
        out.writeObject(serializableVector);

    }
    private void readObject(java.io.ObjectInputStream in)
            throws IOException, ClassNotFoundException{
        numFeatures = in.readInt();
        numComponents = in.readInt();
        weights = ((SerializableVector)in.readObject()).getVector();
    }


}
