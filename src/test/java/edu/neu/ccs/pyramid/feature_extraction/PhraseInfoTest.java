package edu.neu.ccs.pyramid.feature_extraction;

import static org.junit.Assert.*;

public class PhraseInfoTest {
    public static void main(String[] args) {
        test2();
    }

    static void test1(){
        PhraseInfo left = new PhraseInfo("a b c");
        PhraseInfo right = new PhraseInfo("c d");
        System.out.println(PhraseInfo.connect(left,right));
    }

    static void test2(){
        PhraseInfo left = new PhraseInfo("a b c");
        PhraseInfo right = new PhraseInfo("a c d");
        System.out.println(PhraseInfo.connect(left,right));
    }

}