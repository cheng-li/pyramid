package edu.neu.ccs.pyramid.classification.logistic_regression;

import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.VectorView;

import java.io.*;

/**
 * Created by chengli on 12/7/14.
 */
class Weights implements Serializable {
    private static final long serialVersionUID = 1L;
    private int numClasses;
    private int numFeatures;
    /**
     * vector is not serializable
     */
    private transient Vector weightVector;
    /**
     * serialize this array instead
     */
    private double[] serializableWeights;

    Weights(int numClasses, int numFeatures) {
        this.numClasses = numClasses;
        this.numFeatures = numFeatures;
        this.weightVector = new DenseVector((numFeatures + 1)*numClasses);
        this.serializableWeights = new double[(numFeatures + 1)*numClasses];
    }

    int getClassIndex(int parameterIndex){
        return parameterIndex/(numFeatures+1);
    }

    /**
     *
     * @param parameterIndex
     * @return feature index
     * -1 means bias
     */
    int getFeatureIndex(int parameterIndex){
        return parameterIndex - getClassIndex(parameterIndex)*(numFeatures+1) -1;
    }

    /**
     *
     * @return weights for all classes
     */
    Vector getAllWeights() {
        return weightVector;
    }

    int totalSize(){
        return weightVector.size();
    }

    /**
     *
     * @param k class index
     * @return weights for class k, including bias at the beginning
     */
    Vector getWeightsForClass(int k){
        int start = (this.numFeatures+1)*k;
        int length = this.numFeatures +1;
        return new VectorView(this.weightVector,start,length);
    }

    /**
     *
     * @param k
     * @return weights for class k, no bias
     */
    Vector getWeightsWithoutBiasForClass(int k){
        int start = (this.numFeatures+1)*k + 1;
        int length = this.numFeatures;
        return new VectorView(this.weightVector,start,length);
    }

    /**
     *
     * @param k
     * @return bias
     */
    double getBiasForClass(int k){
        int start = (this.numFeatures+1)*k;
        return this.weightVector.get(start);
    }

    private void writeObject(java.io.ObjectOutputStream out)
            throws IOException {
        for (int i=0;i<serializableWeights.length;i++){
            serializableWeights[i] = weightVector.get(i);
        }
        out.writeInt(numClasses);
        out.writeInt(numFeatures);
        out.writeObject(serializableWeights);

    }
    private void readObject(java.io.ObjectInputStream in)
            throws IOException, ClassNotFoundException{
        numClasses = in.readInt();
        numFeatures = in.readInt();
        serializableWeights = (double[])in.readObject();
        weightVector = new DenseVector((numFeatures + 1)*numClasses);
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

    @Override
    public String toString() {
        return "Weights{" +
                "numClasses=" + numClasses +
                ", numFeatures=" + numFeatures +
                ", weightVector=" + weightVector +
                '}';
    }
}
