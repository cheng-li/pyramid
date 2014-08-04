package edu.neu.ccs.pyramid.dataset;


import org.apache.mahout.math.DenseMatrix;
import org.apache.mahout.math.SparseColumnMatrix;

public class SparseDataSetTest {
    public static void main(String[] args) {
        test1();
//        test2();
//        test3();
//        test4();
    }

    static void test1(){
        SparseDataSet dataSet = new SparseDataSet(10,5);
        dataSet.setFeatureValue(1,3,-0.9);
        dataSet.setFeatureValue(1,4,-0.9);
        dataSet.setFeatureValue(1,4,-60.9);
        dataSet.setFeatureValue(7,4,18);
        System.out.println(dataSet);
    }

    static void test2(){
        SparseColumnMatrix matrix = new SparseColumnMatrix(10,5);
        System.out.println(matrix.rowSize());
        System.out.println(matrix.columnSize());
        matrix.set(1, 2, 3);
        matrix.set(1,4,4);
        matrix.set(9,4,100);
        System.out.println(matrix);
        System.out.println(matrix.get(9,4));
    }

    static void test3(){
        SparseColumnMatrix matrix = new SparseColumnMatrix(100000,500000);
        while(true){

        }

    }

    static void test4(){
        DenseMatrix matrix = new DenseMatrix(100000,500000);
    }
}