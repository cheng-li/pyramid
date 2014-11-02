package edu.neu.ccs.pyramid.dataset;


import org.apache.mahout.math.DenseMatrix;
import org.apache.mahout.math.SparseColumnMatrix;

import java.util.stream.IntStream;

public class SparseDataSetTest {
    public static void main(String[] args) {
        test5();
//        test2();
//        test3();
//        test4();
//        test5();
//        test6();
//        test7();
    }

    static void test1(){
        SparseDataSet dataSet = new SparseDataSet(10,5,false);
        dataSet.setFeatureValue(1,3,-0.9);
        dataSet.setFeatureValue(1,4,-0.9);
        dataSet.setFeatureValue(1,4,-60.9);
        dataSet.setFeatureValue(7,4,18);
        System.out.println(dataSet);
        System.out.println(dataSet.getRow(1).print());
        System.out.println(dataSet.getRow(2).print());
        System.out.println(dataSet.getRow(7).print());
        System.out.println(dataSet.getColumn(4).print());
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
        SparseDataSet dataSet = new SparseDataSet(500000,100000,false);
        while(true){

        }

    }

    static void test4(){
        DenseMatrix matrix = new DenseMatrix(100000,500000);
    }

    static void test5(){
        SparseDataSet dataSet = new SparseDataSet(100,10,false);
        IntStream.range(0,100).parallel().
                forEach(i -> IntStream.range(0, 10).parallel().
                        forEach(j -> dataSet.setFeatureValue(i, j, i + j)));
        System.out.println(dataSet);
    }

    static void test6(){
        SparseDataSet dataSet = new SparseDataSet(100,10,false);
        for (int i=0;i<100;i++){
            for (int j=0;j<10;j++){
                dataSet.setFeatureValue(i, j, i + j);
            }
        }
        System.out.println(dataSet);
    }

    static void test7(){
        SparseDataSet dataSet = new SparseDataSet(100,10,false);
        IntStream.range(0,100).parallel().
                forEach(i -> IntStream.range(0, 10).parallel().
                        forEach(j -> dataSet.setFeatureValue(i, j, i + j)));

        SparseDataSet dataSet1 = new SparseDataSet(100,10,false);
        IntStream.range(0,100).parallel().
                forEach(i -> {
                    FeatureRow featureRow = dataSet.getRow(i);
                    IntStream.range(0,10).parallel()
                            .forEach(j-> dataSet1.setFeatureValue(i,j,featureRow.getVector().get(j)));

                });
        System.out.println(dataSet1);


    }
}