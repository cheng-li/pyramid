package edu.neu.ccs.pyramid.clustering.bmm;

import edu.neu.ccs.pyramid.dataset.DataSet;
import edu.neu.ccs.pyramid.dataset.DataSetBuilder;

public class BMMSelectorTest {
    public static void main(String[] args) {
        test1();
    }

    private static void test1(){
        DataSet dataSet = DataSetBuilder.getBuilder()
                .numFeatures(5).numDataPoints(20)
                .dense(true)
                .build();

        for (int i=0;i<5;i++){
            dataSet.setFeatureValue(i,0,1);
        }

        for (int i=5;i<10;i++){
            dataSet.setFeatureValue(i,1,1);
        }

        for (int i=10;i<20;i++){
            dataSet.setFeatureValue(i,1,1);
            dataSet.setFeatureValue(i,2,1);
            dataSet.setFeatureValue(i,3,1);
        }



        System.out.println("dataset = "+dataSet);
        BMM bmm = BMMSelector.select(dataSet,3,10);
        System.out.println(bmm);
        for (int i=0;i<5;i++){
            System.out.println("sample "+i);
            System.out.println(bmm.sample());
        }


    }

}