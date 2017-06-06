package edu.neu.ccs.pyramid.core.clustering.bm;

import edu.neu.ccs.pyramid.core.dataset.DataSet;
import edu.neu.ccs.pyramid.core.dataset.DataSetBuilder;
import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.Vector;

public class BMSelectorTest {
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
        BM bm = BMSelector.select(dataSet,3,10);
        System.out.println(bm);
        for (int i=0;i<5;i++){
            System.out.println("sample "+i);
            System.out.println(bm.sample());
        }

        Vector vector1= new DenseVector(5);
        vector1.set(0,1);

        Vector vector2= new DenseVector(5);
        vector2.set(1,1);

        Vector vector3= new DenseVector(5);
        vector3.set(1,1);
        vector3.set(2,1);
        vector3.set(3,1);

        System.out.println(Math.exp(bm.logProbability(vector1)));
        System.out.println(Math.exp(bm.logProbability(vector2)));
        System.out.println(Math.exp(bm.logProbability(vector3)));


    }

}