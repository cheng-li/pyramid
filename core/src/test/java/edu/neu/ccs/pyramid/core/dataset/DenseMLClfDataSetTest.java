package edu.neu.ccs.pyramid.core.dataset;

public class DenseMLClfDataSetTest {
    public static void main(String[] args) {
        test1();
    }

    static void test1(){
        MultiLabelClfDataSet dataSet = new DenseMLClfDataSet(10,5,false,3);
        dataSet.setFeatureValue(1,3,-0.9);
        dataSet.setFeatureValue(1,4,-0.9);
        dataSet.setFeatureValue(1,4,-60.9);
        dataSet.setFeatureValue(7,4,18);
        dataSet.addLabel(0,1);
        dataSet.addLabel(0,2);
        dataSet.addLabel(1,0);
        System.out.println(dataSet.getMetaInfo());
        System.out.println(dataSet);
    }

}