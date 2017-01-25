package edu.neu.ccs.pyramid.util;

import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.RandomAccessSparseVector;
import org.apache.mahout.math.SequentialAccessSparseVector;
import org.apache.mahout.math.Vector;

/**
 * Created by chengli on 1/3/16.
 */
public class Vectors {

    public static Vector concatenate(Vector vector, double number){
        Vector con = null;
        if (vector instanceof DenseVector){
            con = new DenseVector(vector.size()+1);
        }
        if (vector instanceof RandomAccessSparseVector){
            con = new RandomAccessSparseVector(vector.size()+1);
        }

        if (vector instanceof SequentialAccessSparseVector){
            con = new SequentialAccessSparseVector(vector.size()+1);
        }

        for (Vector.Element nonZeros: vector.nonZeroes()){
            int index = nonZeros.index();
            double value = nonZeros.get();
            con.set(index, value);
        }
        con.set(con.size()-1,number);
        return con;
    }

    public static Vector concatenate(Vector vector, double[] numbers){
        Vector con = null;
        if (vector instanceof DenseVector){
            con = new DenseVector(vector.size()+numbers.length);
        }
        if (vector instanceof RandomAccessSparseVector){
            con = new RandomAccessSparseVector(vector.size()+numbers.length);
        }

        if (vector instanceof SequentialAccessSparseVector){
            con = new SequentialAccessSparseVector(vector.size()+numbers.length);
        }

        for (Vector.Element nonZeros: vector.nonZeroes()){
            int index = nonZeros.index();
            double value = nonZeros.get();
            con.set(index, value);
        }
        for (int i=0;i<numbers.length;i++){
            con.set(i+vector.size(), numbers[i]);
        }
        return con;
    }


    public static double dot(Vector vector1, Vector vector2){
        if (vector1.size()!=vector2.size()){
            throw new IllegalArgumentException("vector1.size()!=vector2.size()");
        }

        boolean vector1Dense = vector1.isDense();
        boolean vector2Dense = vector2.isDense();

        if (vector1Dense&&vector2Dense){
            return dotDenseDense(vector1,vector2);
        } else if (vector1Dense && !vector2Dense){
            return dotDenseSparse(vector1,vector2);
        } else if (!vector1Dense && vector2Dense){
            return dotDenseSparse(vector2,vector1);
        } else {
            throw new UnsupportedOperationException("sparse dot sparse is not supported");
        }

    }

    private static double dotDenseDense(Vector vector1, Vector vector2){
        int size = vector1.size();
        double sum = 0;
        for (int d=0;d<size;d++){
            sum += vector1.getQuick(d)*vector2.getQuick(d);
        }
        return sum;
    }

    private static double dotDenseSparse(Vector denseVector, Vector sparseVector){
        double sum = 0;
        for (Vector.Element element: sparseVector.nonZeroes()){
            int index = element.index();
            double value = element.get();
            sum += value*denseVector.getQuick(index);
        }
        return sum;
    }
}
