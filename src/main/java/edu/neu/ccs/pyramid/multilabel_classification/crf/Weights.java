package edu.neu.ccs.pyramid.multilabel_classification.crf;

import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.VectorView;

import java.io.*;

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

    // if ignore the label pair.
    private boolean featureOnly = false;

    public Weights(int numClasses, int numFeatures, boolean featureOnly) {
        this.numClasses = numClasses;
        this.numFeatures = numFeatures;
        this.numWeightsForFeatures = (numFeatures + 1) * numClasses;
        this.featureOnly = featureOnly;
        if (this.featureOnly) {
            this.numWeightsForLabels = 0;
        }
        else {
            this.numWeightsForLabels = (numClasses * (numClasses-1)/2) * 4;
        }
        this.weightVector = new DenseVector(numWeightsForFeatures + numWeightsForLabels);
        this.serializableWeights = new double[numWeightsForFeatures + numWeightsForLabels];
        System.out.println("numWeightsForFeature: " + numWeightsForFeatures);
        System.out.println("numWeightsForLabels: " + numWeightsForLabels);
    }

    public Weights deepCopy(){
        Weights copy = new Weights(this.numClasses,this.numFeatures,this.featureOnly);
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

    /**
     *
     * @return weights for all label pairs.
     */
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
        int start = numWeightsForFeatures;
        for (int l1=0; l1<numClasses; l1++) {
            for (int l2=l1+1; l2<numClasses; l2++) {
                sb.append("label pair weights: (" +l1 +", " + l2  +")\n");
                sb.append("W(0,0): " + weightVector.get(start) + "\tW(1,0): " + weightVector.get(start+1)+
                "\tW(0,1): " + weightVector.get(start+2) + "\tW(1,1): "+weightVector.get(start+3));
                start += 4;
            }
        }
        sb.append('}');
        return sb.toString();
    }

    private void writeObject(java.io.ObjectOutputStream out)
            throws IOException {
        for (int i=0;i<serializableWeights.length;i++){
            serializableWeights[i] = weightVector.get(i);
        }
        out.writeInt(numClasses);
        out.writeInt(numFeatures);
        out.writeInt(numWeightsForFeatures);
        out.writeInt(numWeightsForLabels);
        out.writeBoolean(featureOnly);
        out.writeObject(serializableWeights);

    }
    private void readObject(java.io.ObjectInputStream in)
            throws IOException, ClassNotFoundException{
        numClasses = in.readInt();
        numFeatures = in.readInt();
        numWeightsForFeatures = in.readInt();
        numWeightsForLabels = in.readInt();
        featureOnly = in.readBoolean();
        serializableWeights = (double[])in.readObject();
        weightVector = new DenseVector(numWeightsForFeatures + numWeightsForLabels);
        for (int i=0;i<serializableWeights.length;i++){
            weightVector.set(i,serializableWeights[i]);
        }
    }

    void serialize(File file) throws Exception{
        File parent = file.getParentFile();
        if (!parent.exists()){
            parent.mkdirs();
        }
        try (
                FileOutputStream fileOutputStream = new FileOutputStream(file);
                BufferedOutputStream bufferedOutputStream = new BufferedOutputStream(fileOutputStream);
                ObjectOutputStream objectOutputStream = new ObjectOutputStream(bufferedOutputStream);
        ){
            objectOutputStream.writeObject(this);
        }
    }

    public static Weights deserialize(File file) throws Exception{
        try(
                FileInputStream fileInputStream = new FileInputStream(file);
                BufferedInputStream bufferedInputStream = new BufferedInputStream(fileInputStream);
                ObjectInputStream objectInputStream = new ObjectInputStream(bufferedInputStream);
        ){
            return (Weights)objectInputStream.readObject();
        }
    }
}
