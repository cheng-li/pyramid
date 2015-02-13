package edu.neu.ccs.pyramid.sentiment_analysis;

import static org.junit.Assert.*;

public class NegationTest {
    public static void main(String[] args) {
        test2();
    }

    private static void test1(){
        String str = "i don't like";
        System.out.println(Negation.removeNegation(str));
    }


    private static void test2(){
        String str = "makes no sense";
        System.out.println(Negation.removeNegation(str));
    }
}