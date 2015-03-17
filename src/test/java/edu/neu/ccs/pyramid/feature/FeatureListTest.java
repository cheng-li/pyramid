package edu.neu.ccs.pyramid.feature;

import static org.junit.Assert.*;

public class FeatureListTest {
    public static void main(String[] args) throws Exception{
        test1();
    }

    private static void test1() throws Exception{
        FeatureList featureList = new FeatureList();
        Ngram ngram1 = new Ngram();
        ngram1.setNgram("ok");
        Ngram ngram2 = new Ngram();
        ngram2.setNgram("ok");
        System.out.println(ngram1.equals(ngram2));
        featureList.add(ngram1);
        featureList.add(ngram2);
        System.out.println(featureList.get(0).equals(featureList.get(1)));
        FeatureList plain = featureList.deepCopy();
        plain.clearInices();
        System.out.println(plain.get(0).equals(plain.get(1)));
        System.out.println(plain.get(0));

    }

}