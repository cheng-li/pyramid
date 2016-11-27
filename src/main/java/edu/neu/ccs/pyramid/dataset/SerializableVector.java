package edu.neu.ccs.pyramid.dataset;

import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.RandomAccessSparseVector;
import org.apache.mahout.math.SequentialAccessSparseVector;
import org.apache.mahout.math.Vector;

import java.io.IOException;
import java.io.Serializable;

/**
 * a wrapper of Vector for serialization
 * Created by chengli on 11/27/16.
 */
public class SerializableVector implements Serializable{
    private static final long serialVersionUID = 1L;
    private Type type;
    private transient Vector vector;

    public SerializableVector(Vector vector) {
        this.vector = vector;
        if (vector instanceof DenseVector){
            type = Type.DENSE;
        } else if (vector instanceof RandomAccessSparseVector){
            type = Type.SPARSE_RANDOM;
        } else {
            type = Type.SPARSE_SEQUENTIAL;
        }
    }


    public Vector getVector() {
        return vector;
    }

    private void writeObject(java.io.ObjectOutputStream out)
            throws IOException {

        out.writeObject(type);
        if (type==Type.DENSE){
            double[] values = new double[vector.size()];
            for (int i=0;i<values.length;i++){
                values[i] = vector.get(i);
            }
            out.writeObject(values);
        } else {
            int numNonZeros = vector.getNumNonZeroElements();
            int[] indices = new int[numNonZeros];
            double[] values = new double[numNonZeros];
            int i=0;
            for (Vector.Element element: vector.nonZeroes()){
                int index = element.index();
                double v = element.get();
                indices[i] = index;
                values[i] = v;
                i += 1;
            }
            out.writeObject(indices);
            out.writeObject(values);
        }
    }

    private void readObject(java.io.ObjectInputStream in)
            throws IOException, ClassNotFoundException{
        type = (Type)in.readObject();
        if (type==Type.DENSE){
            double[] values = (double[])in.readObject();
            vector = new DenseVector(values);
        } else if (type==Type.SPARSE_RANDOM){
            int[] indices = (int[])in.readObject();
            double[] values = (double[])in.readObject();
            vector = new RandomAccessSparseVector();
            for (int i=0;i<indices.length;i++){
                vector.set(indices[i],values[i]);
            }
        } else if (type==Type.SPARSE_SEQUENTIAL){
            int[] indices = (int[])in.readObject();
            double[] values = (double[])in.readObject();
            vector = new SequentialAccessSparseVector();
            for (int i=0;i<indices.length;i++){
                vector.set(indices[i],values[i]);
            }
        }
    }



    private static enum Type{
        DENSE, SPARSE_RANDOM, SPARSE_SEQUENTIAL
    }
}
