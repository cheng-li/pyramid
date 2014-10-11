package edu.neu.ccs.pyramid.dataset;

import static org.junit.Assert.*;

public class LabelTranslatorTest {
    public static void main(String[] args) {
        test1();

    }

    private static void test1(){
        LabelTranslator labelTranslator = LabelTranslator.getBuilder()
                .addExtLabel("a").addExtLabel("b").addExtLabel("c")
                .build();
        System.out.println(labelTranslator);
    }

}