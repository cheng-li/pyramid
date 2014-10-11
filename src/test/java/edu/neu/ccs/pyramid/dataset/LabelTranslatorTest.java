package edu.neu.ccs.pyramid.dataset;

import java.util.HashSet;
import java.util.Set;

import static org.junit.Assert.*;

public class LabelTranslatorTest {
    public static void main(String[] args) {
//        test1();
//        test2();
        test3();
    }

    private static void test1(){
        LabelTranslator labelTranslator = LabelTranslator.getBuilder()
                .addExtLabel("a").addExtLabel("b").addExtLabel("c")
                .build();
        System.out.println(labelTranslator);
    }

    private static void test2(){
        String[] extLabels = {"a","b","c"};
        LabelTranslator labelTranslator = new LabelTranslator(extLabels);
        System.out.println(labelTranslator);
    }

    private static void test3(){
        Set<String> extLabels = new HashSet<>();
        extLabels.add("a");
        extLabels.add("b");
        extLabels.add("c");
        LabelTranslator labelTranslator = new LabelTranslator(extLabels);
        System.out.println(labelTranslator);
    }

}