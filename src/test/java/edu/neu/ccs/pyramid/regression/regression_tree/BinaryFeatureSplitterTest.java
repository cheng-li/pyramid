package edu.neu.ccs.pyramid.regression.regression_tree;

import edu.neu.ccs.pyramid.dataset.SparseDataSet;

public class BinaryFeatureSplitterTest {

    public static void main(String[] args) {
        test1();
        test2();
        test3();

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
        int[] dataAppearance = {0,1,2,3};
        regTreeConfig.setMinDataPerLeaf(1).setDataSet(dataSet).setLabels(labels);
        SplitResult splitResult = BinarySplitter.split(regTreeConfig, dataAppearance, 0);
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
        int[] dataAppearance = {0,1,2,3};
        regTreeConfig.setMinDataPerLeaf(1).setDataSet(dataSet).setLabels(labels);
        SplitResult splitResult = BinarySplitter.split(regTreeConfig, dataAppearance, 0);
        System.out.println(splitResult);
        System.out.println(21-81.0/4);
    }

    static void test3(){
        SparseDataSet dataSet = new SparseDataSet(5,2);
        dataSet.setFeatureValue(0,0,0);
        dataSet.setFeatureValue(1,0,0);
        dataSet.setFeatureValue(2,0,0);
        dataSet.setFeatureValue(3,0,1);
        dataSet.setFeatureValue(4,0,1);
        double[] labels = {1,2,3,3,1};
        RegTreeConfig regTreeConfig = new RegTreeConfig();
        int[] dataAppearance = {0,1,2,3};
        regTreeConfig.setMinDataPerLeaf(2).setDataSet(dataSet).setLabels(labels);
        SplitResult splitResult = BinarySplitter.split(regTreeConfig, dataAppearance, 0);
        System.out.println(splitResult);
        System.out.println(21-81.0/4);
    }



}