package edu.neu.ccs.pyramid.core.regression.linear_regression;

import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.VectorView;

import java.io.*;

/**
 * Created by chengli on 2/18/15.
 */
public class Weights implements Serializable{
    private static final long serialVersionUID = 1L;
    private int numFeatures;
    /**
     * vector is not serializable
     */
    private transient Vector weightVector;
    /**
     * serialize this array instead
     */
    private double[] serializableWeights;

    public Weights(int numFeatures) {
        this.numFeatures = numFeatures;
        this.weightVector = new DenseVector((numFeatures + 1));
        this.serializableWeights = new double[(numFeatures + 1)];
    }

    public Weights(int numFeatures, Vector weightVector) {
        this.numFeatures = numFeatures;
        if (weightVector.size()!=(numFeatures + 1)){
            throw new IllegalArgumentException("weightVector.size()!=(numFeatures + 1)");
        }
        this.weightVector = weightVector;
        this.serializableWeights = new double[(numFeatures + 1)];
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

    /**
     *
     * @return weights , no bias
     */
    public Vector getWeightsWithoutBias(){
        int length = this.numFeatures;
        return new VectorView(this.weightVector,1,length);
    }

    /**
     *
     * @return bias
     */
    public double getBias(){
        return this.weightVector.get(0);
    }

    public void setBias(double bias){
        this.weightVector.set(0,bias);
    }

    public void setWeight(int featureIndex, double weight){
        this.weightVector.set(featureIndex+1,weight);
    }

    private void writeObject(java.io.ObjectOutputStream out)
            throws IOException {
        for (int i=0;i<serializableWeights.length;i++){
            serializableWeights[i] = weightVector.get(i);
        }
        out.writeInt(numFeatures);
        out.writeObject(serializableWeights);

    }
    private void readObject(java.io.ObjectInputStream in)
            throws IOException, ClassNotFoundException{
        numFeatures = in.readInt();
        serializableWeights = (double[])in.readObject();
        weightVector = new DenseVector((numFeatures + 1));
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
                ", numFeatures=" + numFeatures +
                ", weightVector=" + weightVector +
                '}';
    }
}
