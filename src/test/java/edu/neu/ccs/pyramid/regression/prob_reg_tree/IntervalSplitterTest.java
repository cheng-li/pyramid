package edu.neu.ccs.pyramid.regression.prob_reg_tree;

import edu.neu.ccs.pyramid.dataset.SparseDataSet;


import static org.junit.Assert.*;

public class IntervalSplitterTest {
    public static void main(String[] args) {
//        test1();
//        test2();
//        test3();
//        test4();
        test5();
    }


    private static void test1(){
        SparseDataSet dataSet = new SparseDataSet(5,2);
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
        SparseDataSet dataSet = new SparseDataSet(5,2);
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
        SparseDataSet dataSet = new SparseDataSet(5,2);
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
        SparseDataSet dataSet = new SparseDataSet(8,1);
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
        SparseDataSet dataSet = new SparseDataSet(8,1);
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


    
}