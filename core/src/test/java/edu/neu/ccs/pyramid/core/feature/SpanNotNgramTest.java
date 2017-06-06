package edu.neu.ccs.pyramid.core.feature;

public class SpanNotNgramTest {
    public static void main(String[] args) {
        test1();
    }


    private static void test1(){
        Ngram ngram = new Ngram();
        ngram.setInOrder(true);
        ngram.setNgram("skip movi");
        ngram.setField("body");
        ngram.setSlop(0);

        System.out.println(SpanNotNgram.breakBigram(ngram));
    }

}