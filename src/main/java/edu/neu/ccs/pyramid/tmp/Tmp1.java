package edu.neu.ccs.pyramid.tmp;

import org.apache.commons.lang3.time.StopWatch;
import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.RandomAccessSparseVector;
import org.apache.mahout.math.Vector;

import java.util.Iterator;

/**
 * Created by chengli on 8/4/14.
 */
public class Tmp1 {
    public static void main(String[] args) {
        test7();
    }

    static void test1(){
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
    static void test2(){
        RandomAccessSparseVector vector = new RandomAccessSparseVector(50000);
        StopWatch watch = new StopWatch();
        watch.start();
        for (int i=0;i<vector.size();i++){
            vector.set(i,1);
        }
        System.out.println(watch);
        DenseVector denseVector = new DenseVector(vector);
        System.out.println(watch);

    }

    static void test3(){
        RandomAccessSparseVector vector = new RandomAccessSparseVector(50000);
        StopWatch watch = new StopWatch();
        watch.start();
        for (int i=0;i<100;i++){
            vector.set(i,Math.random());
        }
        System.out.println(watch);
        DenseVector denseVector = new DenseVector(vector);
        System.out.println(watch);

    }

    static void test4(){

        StopWatch watch = new StopWatch();
        watch.start();
        DenseVector vector = new DenseVector(50000);
        System.out.println(watch);


    }

    static void test5(){
        RandomAccessSparseVector vector = new RandomAccessSparseVector(50000);
        StopWatch watch = new StopWatch();
        watch.start();
        for (int i=0;i<100;i++){
            vector.set(i,1);
        }
        System.out.println(watch);
        DenseVector denseVector = new DenseVector(50000);
        Iterator<Vector.Element> iterator = vector.iterateNonZero();
        while(iterator.hasNext()){
            Vector.Element element = iterator.next();
            int index = element.index();
            double value = element.get();
            denseVector.setQuick(index,value);
        }
        System.out.println(watch);
    }

    static void test6(){
        StopWatch watch = new StopWatch();
        watch.start();
        double[] arr = new double[50000];
        System.out.println(watch);
    }

    static void test7(){
        RandomAccessSparseVector vector = new RandomAccessSparseVector(50000);
        StopWatch watch = new StopWatch();
        watch.start();
        for (int i=0;i<100;i++){
            vector.set(i,1);
        }
        System.out.println(watch);
        double[] arr = new double[50000];
        Iterator<Vector.Element> iterator = vector.iterateNonZero();
        while(iterator.hasNext()){
            Vector.Element element = iterator.next();
            int index = element.index();
            double value = element.get();
            arr[index]=value;
        }
        System.out.println(watch);
    }
}
