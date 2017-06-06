package edu.neu.ccs.pyramid.core.classification.logistic_regression;

import org.apache.commons.math3.distribution.UniformRealDistribution;
import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.VectorView;

import java.io.*;
import java.util.ArrayList;
import java.util.List;

/**
 * Created by chengli on 12/7/14.
 */
public class Weights implements Serializable {
    private static final long serialVersionUID = 2L;
    private int numClasses;
    private int numFeatures;
    /**
     * vector is not serializable
     */
    private transient Vector weightVector;


    public Weights(int numClasses, int numFeatures, boolean random) {
        if (random) {
            this.numClasses = numClasses;
            this.numFeatures = numFeatures;
            this.weightVector = new DenseVector((numFeatures + 1)*numClasses);
            UniformRealDistribution uniform = new UniformRealDistribution(-0.5,0.5);
            for (int i=0; i<weightVector.size(); i++) {
                double p = uniform.sample();
                weightVector.set(i,p);
            }
        } else {
            this.numClasses = numClasses;
            this.numFeatures = numFeatures;
            this.weightVector = new DenseVector((numFeatures + 1)*numClasses);
        }

    }

    public Weights(int numClasses, int numFeatures) {
        this.numClasses = numClasses;
        this.numFeatures = numFeatures;
        this.weightVector = new DenseVector((numFeatures + 1)*numClasses);
    }

    public Weights(int numClasses, int numFeatures, Vector weightVector) {
        this.numClasses = numClasses;
        this.numFeatures = numFeatures;
        if (weightVector.size()!=(numFeatures + 1)*numClasses){
            throw new IllegalArgumentException("weightVector.size()!=(numFeatures + 1)*numClasses");
        }
        this.weightVector = weightVector;
    }

    public void setWeightVector(Vector weightVector) {
        if (weightVector.size()!=(numFeatures + 1)*numClasses){
            throw new IllegalArgumentException("weightVector.size()!=(numFeatures + 1)*numClasses");
        }
        this.weightVector = weightVector;
    }

    public Weights deepCopy(){
        Weights copy = new Weights(this.numClasses,numFeatures);
        copy.weightVector = new DenseVector(this.weightVector);
        return copy;
    }

    public int getClassIndex(int parameterIndex){
        return parameterIndex/(numFeatures+1);
    }

    /**
     *
     * @param parameterIndex
     * @return feature index
     * -1 means bias
     */
    public int getFeatureIndex(int parameterIndex){
        return parameterIndex - getClassIndex(parameterIndex)*(numFeatures+1) -1;
    }

    public List<Integer> getAllBiasPositions(){
        List<Integer> list = new ArrayList<>();
        for (int k=0;k<numClasses;k++){
            list.add((this.numFeatures+1)*k);
        }
        return list;
    }

    /**
     *
     * @return weights for all classes
     */
    public Vector getAllWeights() {
        return weightVector;
    }

    public int totalSize(){
        return weightVector.size();
    }

    /**
     *
     * @param k class index
     * @return weights for class k, including bias at the beginning
     */
    public Vector getWeightsForClass(int k){
        if (k>=numClasses){
            throw new IllegalArgumentException("out of bound");
        }
        int start = (this.numFeatures+1)*k;
        int length = this.numFeatures +1;
        return new VectorView(this.weightVector,start,length);
    }

    /**
     *
     * @param k
     * @return weights for class k, no bias
     */
    public Vector getWeightsWithoutBiasForClass(int k){
        if (k>=numClasses){
            throw new IllegalArgumentException("out of bound");
        }
        int start = (this.numFeatures+1)*k + 1;
        int length = this.numFeatures;
        return new VectorView(this.weightVector,start,length);
    }

    /**
     *
     * @param k
     * @return bias
     */
    public double getBiasForClass(int k){
        if (k>=numClasses){
            throw new IllegalArgumentException("out of bound");
        }
        int start = (this.numFeatures+1)*k;
        return this.weightVector.get(start);
    }

    private void writeObject(java.io.ObjectOutputStream out)
            throws IOException {
        out.writeInt(numClasses);
        out.writeInt(numFeatures);
        int numNonZeros = weightVector.getNumNonZeroElements();
        int[] indices = new int[numNonZeros];
        double[] values = new double[numNonZeros];
        int i=0;
        for (Vector.Element element: weightVector.nonZeroes()){
            int index = element.index();
            double v = element.get();
            indices[i] = index;
            values[i] = v;
            i += 1;
        }
        out.writeObject(indices);
        out.writeObject(values);

    }
    private void readObject(java.io.ObjectInputStream in)
            throws IOException, ClassNotFoundException{
        numClasses = in.readInt();
        numFeatures = in.readInt();
        int[] indices = (int[])in.readObject();
        double[] values = (double[])in.readObject();
        weightVector = new DenseVector((numFeatures + 1)*numClasses);
        for (int i=0;i<indices.length;i++){
            weightVector.set(indices[i],values[i]);
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
