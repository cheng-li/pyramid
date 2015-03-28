package edu.neu.ccs.pyramid.feature;

import static org.junit.Assert.*;

public class NgramTest {
    public static void main(String[] args) {
        test1();

    }

    private static void test1(){
        Ngram ngram = new Ngram();
        ngram.setInOrder(true);
        ngram.setNgram("skip movi");
        ngram.setField("body");
        ngram.setSlop(0);

        Ngram ngram2 = new Ngram();
        ngram2.setInOrder(false);
        ngram2.setNgram("skip movi");
        ngram2.setField("body");
        ngram2.setSlop(0);

        System.out.println(ngram.equals(ngram2));
    }

}