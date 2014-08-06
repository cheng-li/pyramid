package edu.neu.ccs.pyramid.regression.regression_tree;

import edu.neu.ccs.pyramid.dataset.SparseDataSet;

import static org.junit.Assert.*;

public class IntervalSplitterTest {
    public static void main(String[] args) {
//        test1();
        test2();
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

}