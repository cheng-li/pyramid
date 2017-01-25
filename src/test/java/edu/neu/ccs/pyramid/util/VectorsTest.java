package edu.neu.ccs.pyramid.util;

import junit.framework.TestCase;
import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.Vector;

/**
 * Created by chengli on 1/21/17.
 */
public class VectorsTest {
    public static void main(String[] args) {
        double[] d = {1,2,5};
        Vector v = new DenseVector(d);
        System.out.println(Vectors.concatenate(v,4.5));

        test2();
    }

    private static void test2(){
        double[] d = {1,2,3};
        Vector v = new DenseVector(d);
        double[] a = {7,4,5,6};
        System.out.println(Vectors.concatenate(v,a));
    }

}