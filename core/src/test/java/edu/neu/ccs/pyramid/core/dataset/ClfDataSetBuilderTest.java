package edu.neu.ccs.pyramid.core.dataset;

public class ClfDataSetBuilderTest {
    public static void main(String[] args) {
        test1();
    }

    private static void test1(){
        ClfDataSet dataSet = ClfDataSetBuilder.getBuilder()
                .numDataPoints(10).numFeatures(5).dense(false)
                .missingValue(true).numClasses(3).build();
        System.out.println(dataSet.getMetaInfo());
    }

}