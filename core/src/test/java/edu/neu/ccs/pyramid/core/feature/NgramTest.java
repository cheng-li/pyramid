package edu.neu.ccs.pyramid.core.feature;

public class NgramTest {
    public static void main(String[] args) {
        test3();

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

    private static void test2(){
        Ngram ngram = new Ngram();
        ngram.setInOrder(true);
        ngram.setNgram("skip movi");
        ngram.setField("body");
        ngram.setSlop(0);

        System.out.println(ngram.contains("skip"));
        System.out.println(ngram.contains("a"));
        System.out.println(ngram.contains(""));
    }


    private static void test3(){
        Ngram ngram = new Ngram();
        ngram.setInOrder(true);
        ngram.setNgram("skip movi");
        ngram.setField("body");
        ngram.setSlop(0);


        Ngram ngram2 = new Ngram();
        ngram2.setInOrder(true);
        ngram2.setNgram("skip it");
        ngram2.setField("body");
        ngram2.setSlop(0);

        Ngram ngram3 = new Ngram();
        ngram3.setInOrder(true);
        ngram3.setNgram("like it");
        ngram3.setField("body");
        ngram3.setSlop(0);

        System.out.println(Ngram.overlap(ngram,ngram2));
        System.out.println(Ngram.overlap(ngram2,ngram3));
        System.out.println(Ngram.overlap(ngram,ngram3));

    }

}