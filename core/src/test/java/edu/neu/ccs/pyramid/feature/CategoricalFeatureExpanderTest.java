package edu.neu.ccs.pyramid.feature;

import java.util.List;

public class CategoricalFeatureExpanderTest {
    public static void main(String[] args) {
test1();
    }

    private static void test1(){
        CategoricalFeatureExpander expander = new CategoricalFeatureExpander();
        expander.setVariableName("color");
        expander.setStart(10);
        expander.addCategory("red");
        expander.addCategory("blue");
        expander.addCategory("yellow");
        expander.putSetting("source","field");
        List<CategoricalFeature> features = expander.expand();
        System.out.println(features);
    }

}