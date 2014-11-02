package edu.neu.ccs.pyramid.regression.regression_tree;

import edu.neu.ccs.pyramid.dataset.SparseDataSet;
import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.Vector;


import java.util.List;

public class IntervalSplitterTest {
    public static void main(String[] args) {
//        test1();
//        test2();
//        test3();
//        test4();
//        test5();
//        test6();
//        test7();
//        test8();
//    test9();
//        test10();
//        test11();
        test12();
    }


    private static void test1(){
        SparseDataSet dataSet = new SparseDataSet(5,2,false);
        dataSet.setFeatureValue(0,0,0);
        dataSet.setFeatureValue(1,0,0);
        dataSet.setFeatureValue(2,0,0);
        dataSet.setFeatureValue(3,0,1);
        dataSet.setFeatureValue(4,0,1);
        double[] labels = {1,2,3,3,1};
        RegTreeConfig regTreeConfig = new RegTreeConfig();
        double[] probs = {1,1,1,1,0};
        regTreeConfig.setMinDataPerLeaf(1)
                .setNumSplitIntervals(2);
        System.out.println(IntervalSplitter.split(regTreeConfig, dataSet, labels, probs, 0));
    }

    private static void test2(){
        SparseDataSet dataSet = new SparseDataSet(5,2,false);
        dataSet.setFeatureValue(0,0,0);
        dataSet.setFeatureValue(1,0,0);
        dataSet.setFeatureValue(2,0,0);
        dataSet.setFeatureValue(3,0,1);
        dataSet.setFeatureValue(4,0,1);
        double[] labels = {1,2,3,3,1};
        RegTreeConfig regTreeConfig = new RegTreeConfig();
        double[] probs = {1,1,1,1,0};
        regTreeConfig.setMinDataPerLeaf(1)
                .setNumSplitIntervals(200);
        System.out.println(IntervalSplitter.split(regTreeConfig, dataSet, labels, probs, 0));
    }

    private static void test3(){
        SparseDataSet dataSet = new SparseDataSet(5,2,false);
        dataSet.setFeatureValue(0,0,0);
        dataSet.setFeatureValue(1,0,0);
        dataSet.setFeatureValue(2,0,0);
        dataSet.setFeatureValue(3,0,1);
        dataSet.setFeatureValue(4,0,1);
        double[] labels = {1,2,3,3,1};
        RegTreeConfig regTreeConfig = new RegTreeConfig();
        double[] probs = {1,1,1,1,0};
        regTreeConfig.setMinDataPerLeaf(2)
                .setNumSplitIntervals(200);
        System.out.println(IntervalSplitter.split(regTreeConfig,dataSet,labels,probs,0));
    }

    private static void test4(){
        SparseDataSet dataSet = new SparseDataSet(8,1,false);
        dataSet.setFeatureValue(0,0,0);
        dataSet.setFeatureValue(1,0,0);
        dataSet.setFeatureValue(2,0,0);
        dataSet.setFeatureValue(3,0,1);
        dataSet.setFeatureValue(4,0,1);
        dataSet.setFeatureValue(5,0,2);
        dataSet.setFeatureValue(6,0,2);
        dataSet.setFeatureValue(6,0,3);
        double[] labels = {1,2,3,3,1,5,5,5};
        RegTreeConfig regTreeConfig = new RegTreeConfig();
        double[] probs = {1,1,1,1,1,1,1,1};
        regTreeConfig.setMinDataPerLeaf(2).setNumSplitIntervals(4);
        System.out.println(IntervalSplitter.split(regTreeConfig, dataSet, labels, probs, 0));
    }

    static void test5(){
        SparseDataSet dataSet = new SparseDataSet(8,1,false);
        dataSet.setFeatureValue(0,0,0);
        dataSet.setFeatureValue(1,0,0);
        dataSet.setFeatureValue(2,0,0);
        dataSet.setFeatureValue(3,0,1);
        dataSet.setFeatureValue(4,0,1);
        dataSet.setFeatureValue(5,0,2);
        dataSet.setFeatureValue(6,0,2);
        dataSet.setFeatureValue(6,0,3);
        double[] labels = {1,2,3,3,1,5,5,5};
        RegTreeConfig regTreeConfig = new RegTreeConfig();
        double[] probs = {1,1,1,1,1,1,1,1};
        regTreeConfig.setMinDataPerLeaf(5).setNumSplitIntervals(4);
        System.out.println(IntervalSplitter.split(regTreeConfig, dataSet, labels, probs, 0));
    }

    static void test6(){
        RegTreeConfig regTreeConfig = new RegTreeConfig().setNumSplitIntervals(4);
        Vector vector = new DenseVector(4);
        vector.set(0,0);
        vector.set(1,1);
        vector.set(2,2);
        vector.set(3,3);
        double[] probs = {1,0.5,0.2,0.6};
        double[] labels = {1,2,3,4};
        System.out.println(IntervalSplitter.generateIntervalsWithMissingValue(regTreeConfig, vector, probs, labels));

    }

    static void test7(){
        RegTreeConfig regTreeConfig = new RegTreeConfig().setNumSplitIntervals(4);
        Vector vector = new DenseVector(4);
        vector.set(0,0);
        vector.set(1,1);
        vector.set(2,2);
        vector.set(3,3);
        double[] probs = {0,0.5,0.2,0.6};
        double[] labels = {1,2,3,4};
        List<Interval> intervals = IntervalSplitter.generateIntervalsWithMissingValue(regTreeConfig, vector, probs, labels);
        System.out.println(intervals);
        System.out.println(IntervalSplitter.compress(intervals));

    }

    static void test8(){
        RegTreeConfig regTreeConfig = new RegTreeConfig().setNumSplitIntervals(4);
        Vector vector = new DenseVector(4);
        vector.set(0,0);
        vector.set(1,1);
        vector.set(2,2);
        vector.set(3,3);
        double[] probs = {1,0.5,0,0.6};
        double[] labels = {1,2,3,4};
        List<Interval> intervals = IntervalSplitter.generateIntervalsWithMissingValue(regTreeConfig, vector, probs, labels);
        System.out.println(intervals);
        System.out.println(IntervalSplitter.compress(intervals));

    }

    static void test9(){
        RegTreeConfig regTreeConfig = new RegTreeConfig().setNumSplitIntervals(2);
        Vector vector = new DenseVector(4);
        vector.set(0,0);
        vector.set(1,1);
        vector.set(2,Double.NaN);
        vector.set(3,3);
        double[] probs = {1,0.5,1,0.6};
        double[] labels = {1,2,3,4};
        List<Interval> intervals = IntervalSplitter.generateIntervalsWithMissingValue(regTreeConfig, vector, probs, labels);
        System.out.println(intervals);
        System.out.println(IntervalSplitter.compress(intervals));
        System.out.println(1.5/(1.5+0.6));
        System.out.println(1+1+3*1.5/(1.5+0.6));

    }

    static void test10(){
        RegTreeConfig regTreeConfig = new RegTreeConfig().setNumSplitIntervals(2);
        Vector vector = new DenseVector(4);
        vector.set(0,Double.NaN);
        vector.set(1,1);
        vector.set(2,2);
        vector.set(3,3);
        double[] probs = {1,0.5,1,0.6};
        double[] labels = {1,2,3,4};
        List<Interval> intervals = IntervalSplitter.generateIntervalsWithMissingValue(regTreeConfig, vector, probs, labels);
        System.out.println(intervals);
        System.out.println(IntervalSplitter.compress(intervals));

    }

    static void test11(){
        RegTreeConfig regTreeConfig = new RegTreeConfig().setNumSplitIntervals(2);
        Vector vector = new DenseVector(4);
        vector.set(0,Double.NaN);
        vector.set(1,1);
        vector.set(2,Double.NaN);
        vector.set(3,3);
        double[] probs = {1,0.5,1,0.6};
        double[] labels = {1,2,3,4};
        List<Interval> intervals = IntervalSplitter.generateIntervalsWithMissingValue(regTreeConfig, vector, probs, labels);
        System.out.println(intervals);
        System.out.println(IntervalSplitter.compress(intervals));

    }

    static void test12(){
        RegTreeConfig regTreeConfig = new RegTreeConfig().setNumSplitIntervals(2);
        Vector vector = new DenseVector(4);
        vector.set(0,Double.NaN);
        vector.set(1,Double.NaN);
        vector.set(2,Double.NaN);
        vector.set(3,3);
        double[] probs = {1,0.5,1,0.6};
        double[] labels = {1,2,3,4};
        List<Interval> intervals = IntervalSplitter.generateIntervalsWithMissingValue(regTreeConfig, vector, probs, labels);
        System.out.println(intervals);
        System.out.println(IntervalSplitter.compress(intervals));

    }



}