package edu.neu.ccs.pyramid.dataset;

import static org.junit.Assert.*;

public class DenseClfDataSetTest {
    public static void main(String[] args) {
        test1();
    }

    static void test1(){
        ClfDataSet dataSet = new DenseClfDataSet(10,5);
        dataSet.setFeatureValue(1,3,-0.9);
        dataSet.setFeatureValue(1,4,-0.9);
        dataSet.setFeatureValue(1,4,-60.9);
        dataSet.setFeatureValue(7,4,18);
        dataSet.setLabel(0,4);
        dataSet.setLabel(4,30);
        System.out.println(dataSet);
    }

}