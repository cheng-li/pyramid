package edu.neu.ccs.pyramid.core.feature;

import java.util.ArrayList;
import java.util.List;

public class FeatureTest {
    public static void main(String[] args) {
        test8();

    }

    private static void test1(){
        Feature feature1 = new Feature();
        Feature feature2 = new Feature();
        System.out.println(feature1.equals(feature2));

    }

    private static void test2(){
        Feature feature1 = new Feature();
        feature1.setIndex(0);
        Feature feature2 = new Feature();
        System.out.println(feature1.equals(feature2));
    }

    private static void test3(){
        Feature feature1 = new Feature();
        feature1.setIndex(0);
        Feature feature2 = new Feature();
        feature2.setIndex(0);
        System.out.println(feature1.equals(feature2));
    }


    private static void test4(){
        Feature feature1 = new Feature();
        feature1.setIndex(0);
        Feature feature2 = new Feature();
        feature2.setIndex(1);
        System.out.println(feature1.equals(feature2));
    }

    private static void test5(){
        Feature feature1 = new Feature();
        feature1.setName("a");
        Feature feature2 = new Feature();
        System.out.println(feature1.equals(feature2));
    }

    private static void test6(){
        Feature feature1 = new Feature();
        feature1.setName("a");
        feature1.setIndex(0);
        Feature feature2 = new Feature();
        feature2.setName("a");
        feature2.setIndex(1);
        System.out.println(feature1.equals(feature2));
        feature1.clearIndex();
        feature2.clearIndex();
        System.out.println(feature1.equals(feature2));
    }

    private static void test7(){
        Feature ngram = new Ngram();
        Feature cate = new CategoricalFeature();
        System.out.println(ngram.equals(cate));
        System.out.println(ngram.getClass());
    }

    private static void test8(){
        Feature feature1 = new Feature();
        feature1.setName("a");
        feature1.setIndex(0);
        Feature feature2 = new Feature();
        feature2.setName("a");
        feature2.setIndex(0);
        List<Feature> features = new ArrayList<>();
        features.add(feature1);
        System.out.println(features.contains(feature2));
    }

}