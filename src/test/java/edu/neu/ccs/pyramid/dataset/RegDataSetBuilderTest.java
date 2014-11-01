package edu.neu.ccs.pyramid.dataset;

import static org.junit.Assert.*;

public class RegDataSetBuilderTest {
    public static void main(String[] args) {
        test1();
    }

    private static void test1(){
        RegDataSet dataSet = RegDataSetBuilder.getBuilder()
                .numDataPoints(2).numFeatures(5).dense(false)
                .missingValue(true).build();
        System.out.println(dataSet.getMetaInfo());
    }

}