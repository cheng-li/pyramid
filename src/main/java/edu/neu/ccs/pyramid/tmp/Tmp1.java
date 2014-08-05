package edu.neu.ccs.pyramid.tmp;

import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.RandomAccessSparseVector;
import org.apache.mahout.math.Vector;

/**
 * Created by chengli on 8/4/14.
 */
public class Tmp1 {
    public static void main(String[] args) {
        RandomAccessSparseVector vector = new RandomAccessSparseVector(10);
        vector.set(1,10);
        vector.set(3,3);
        vector.set(4,9.7);
        vector.set(4,0);
        vector.set(5,3);
        System.out.println(vector.getIteratorAdvanceCost());
        System.out.println(vector.getLookupCost());
        for (Vector.Element elment: vector.all()){
            System.out.println(elment.index()+" "+elment.get());
        }
        for (Vector.Element element: vector.nonZeroes()){
            System.out.println(element.index()+" "+element.get());
        }
        System.out.println();
        DenseVector denseVector = new DenseVector(vector);
        System.out.println(denseVector.asFormatString());
    }
}
