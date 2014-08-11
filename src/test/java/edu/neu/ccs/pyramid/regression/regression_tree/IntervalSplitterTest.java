package edu.neu.ccs.pyramid.regression.regression_tree;

import edu.neu.ccs.pyramid.dataset.SparseDataSet;

import static org.junit.Assert.*;

public class IntervalSplitterTest {
    public static void main(String[] args) {

        test6();
    }
    static void test1(){
        System.out.println(Math.ceil(1.5));
        System.out.println((int)Math.ceil(1.5));
        System.out.println(Math.ceil(-1.5));
        System.out.println((int)Math.ceil(-1.5));
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
        regTreeConfig.setMinDataPerLeaf(1).setDataSet(dataSet)
                .setLabels(labels).setNumSplitIntervals(2);
        System.out.println(IntervalSplitter.split(regTreeConfig,dataAppearance,0));


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
        regTreeConfig.setMinDataPerLeaf(1).setDataSet(dataSet)
                .setLabels(labels).setNumSplitIntervals(200);
        System.out.println(IntervalSplitter.split(regTreeConfig,dataAppearance,0));


    }

    static void test4(){
        SparseDataSet dataSet = new SparseDataSet(5,2);
        dataSet.setFeatureValue(0,0,0);
        dataSet.setFeatureValue(1,0,0);
        dataSet.setFeatureValue(2,0,0);
        dataSet.setFeatureValue(3,0,1);
        dataSet.setFeatureValue(4,0,1);
        double[] labels = {1,2,3,3,1};
        RegTreeConfig regTreeConfig = new RegTreeConfig();
        int[] dataAppearance = {0,1,2,3};
        regTreeConfig.setMinDataPerLeaf(2).setDataSet(dataSet)
                .setLabels(labels).setNumSplitIntervals(200);
        System.out.println(IntervalSplitter.split(regTreeConfig,dataAppearance,0));


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
        int[] dataAppearance = {0,1,2,3,4,5,6,7};
        regTreeConfig.setMinDataPerLeaf(2).setDataSet(dataSet)
                .setLabels(labels).setNumSplitIntervals(4);
        System.out.println(IntervalSplitter.split(regTreeConfig,dataAppearance,0));


    }

    /**
     * test empty
     */
    static void test6(){
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
        int[] dataAppearance = {0,1,2,3,4,5,6,7};
        regTreeConfig.setMinDataPerLeaf(5).setDataSet(dataSet)
                .setLabels(labels).setNumSplitIntervals(4);
        System.out.println(IntervalSplitter.split(regTreeConfig,dataAppearance,0));


    }

}