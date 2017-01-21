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
    }

}