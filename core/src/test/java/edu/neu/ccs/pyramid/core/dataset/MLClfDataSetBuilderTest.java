package edu.neu.ccs.pyramid.core.dataset;

public class MLClfDataSetBuilderTest {
    public static void main(String[] args) {
        test1();
    }

    private static void test1(){
        MultiLabelClfDataSet dataSet = MLClfDataSetBuilder.getBuilder()
                .numDataPoints(10).numFeatures(3).numClasses(4).
                        dense(true).missingValue(false).build();
        System.out.println(dataSet.getMetaInfo());
    }

}