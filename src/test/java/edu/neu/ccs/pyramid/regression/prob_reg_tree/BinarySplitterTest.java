package edu.neu.ccs.pyramid.regression.prob_reg_tree;

import edu.neu.ccs.pyramid.dataset.SparseDataSet;


import static org.junit.Assert.*;

public class BinarySplitterTest {
    public static void main(String[] args) {
        test1();
        test2();

    }

    static void test1(){
        SparseDataSet dataSet = new SparseDataSet(5,2);
        dataSet.setFeatureValue(0,0,0);
        dataSet.setFeatureValue(1,0,0);
        dataSet.setFeatureValue(2,0,0);
        dataSet.setFeatureValue(3,0,1);
        dataSet.setFeatureValue(4,0,1);
        double[] labels = {1,2,3,2,1};
        RegTreeConfig regTreeConfig = new RegTreeConfig();
        double[] probs = {1,1,1,1,0};
        regTreeConfig.setMinDataPerLeaf(1);
        SplitResult splitResult = BinarySplitter.split(regTreeConfig, dataSet, labels, probs, 0).get();
        System.out.println(splitResult);
    }

    static void test2(){
        SparseDataSet dataSet = new SparseDataSet(5,2);
        dataSet.setFeatureValue(0,0,0);
        dataSet.setFeatureValue(1,0,0);
        dataSet.setFeatureValue(2,0,0);
        dataSet.setFeatureValue(3,0,1);
        dataSet.setFeatureValue(4,0,1);
        double[] labels = {1,2,3,3,1};
        RegTreeConfig regTreeConfig = new RegTreeConfig();
        double[] probs = {1,1,1,1,0};
        regTreeConfig.setMinDataPerLeaf(1);
        SplitResult splitResult = BinarySplitter.split(regTreeConfig, dataSet, labels, probs, 0).get();
        System.out.println(splitResult);
        System.out.println(21-81.0/4);
    }

//    /**
//     * test empty
//     */
//    static void test3(){
//        SparseDataSet dataSet = new SparseDataSet(5,2);
//        dataSet.setFeatureValue(0,0,0);
//        dataSet.setFeatureValue(1,0,0);
//        dataSet.setFeatureValue(2,0,0);
//        dataSet.setFeatureValue(3,0,1);
//        dataSet.setFeatureValue(4,0,1);
//        double[] labels = {1,2,3,3,1};
//        RegTreeConfig regTreeConfig = new RegTreeConfig();
//        int[] dataAppearance = {0,1,2,3};
//        regTreeConfig.setMinDataPerLeaf(2);
//        System.out.println(edu.neu.ccs.pyramid.regression.regression_tree.BinarySplitter.split(regTreeConfig, dataSet, labels, dataAppearance, 0));
//        System.out.println(BinarySplitter.split(regTreeConfig, dataSet, labels, dataAppearance, 0).isPresent());
//
//    }

}