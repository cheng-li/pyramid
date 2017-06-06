package edu.neu.ccs.pyramid.core.classification.logistic_regression;

import edu.neu.ccs.pyramid.core.configuration.Config;
import org.apache.mahout.math.Vector;

import java.io.File;

public class WeightsTest {
    private static final Config config = new Config("config/local.properties");
    private static final String DATASETS = config.getString("input.datasets");
    private static final String TMP = config.getString("output.tmp");

    public static void main(String[] args) throws Exception{
//        test1();
//        test2();
        test3();
    }

    private static void test1() throws Exception{
        Weights weights = new Weights(2,5);
        Vector vector = weights.getAllWeights();
        vector.set(0,100);
        vector.set(1,4);
        vector.set(4,-60);
        vector.set(5,9);
        vector.set(6,99);
        vector.set(8,-8.5);
        vector.set(11,45);
        weights.serialize(new File(TMP,"weights"));
        System.out.println("ok");

    }

    private static void test2() throws Exception{
        Weights weights = Weights.deserialize(new File(TMP,"weights"));
        System.out.println(weights);
        System.out.println(weights.getWeightsForClass(1));
        System.out.println(weights.getWeightsWithoutBiasForClass(1));
        System.out.println(weights.getBiasForClass(0));
        System.out.println(weights.getBiasForClass(1));
    }

    private static void test3() throws Exception{
        Weights weights = new Weights(3,5);
        System.out.println(weights.getClassIndex(0));
        System.out.println(weights.getClassIndex(5));
        System.out.println(weights.getClassIndex(6));
        System.out.println(weights.getClassIndex(11));
        System.out.println(weights.getClassIndex(12));
        System.out.println(weights.getClassIndex(17));
        System.out.println(weights.getFeatureIndex(0));
        System.out.println(weights.getFeatureIndex(1));
        System.out.println(weights.getFeatureIndex(5));
        System.out.println(weights.getFeatureIndex(6));
        System.out.println(weights.getFeatureIndex(7));
        System.out.println(weights.getFeatureIndex(11));
    }

}