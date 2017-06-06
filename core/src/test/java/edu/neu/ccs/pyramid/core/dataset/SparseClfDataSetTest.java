package edu.neu.ccs.pyramid.core.dataset;

public class SparseClfDataSetTest {
    public static void main(String[] args) {
        test2();
    }

    static void test1(){
        ClfDataSet dataSet = new SparseClfDataSet(10,5,false,2);
        dataSet.setFeatureValue(1,3,-0.9);
        dataSet.setFeatureValue(1,4,-0.9);
        dataSet.setFeatureValue(1,4,-60.9);
        dataSet.setFeatureValue(7,4,18);
        dataSet.setLabel(0,4);
        dataSet.setLabel(4,30);
        System.out.println(dataSet);
    }

    static void test2(){
        ClfDataSet dataSet = new SparseClfDataSet(10,5,false,2);
        dataSet.setFeatureValue(1,3,-0.9);
        dataSet.setFeatureValue(1,4,-0.9);
        dataSet.setFeatureValue(1,4,-60.9);
        dataSet.setFeatureValue(7,4,18);
        dataSet.setLabel(0,0);
        dataSet.setLabel(4,1);
        System.out.println(dataSet.getMetaInfo());
        System.out.println(dataSet);
    }

}