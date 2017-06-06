package edu.neu.ccs.pyramid.util;

import org.apache.mahout.math.Vector;

/**
 * Created by chengli on 1/3/16.
 */
public class Vectors {
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
