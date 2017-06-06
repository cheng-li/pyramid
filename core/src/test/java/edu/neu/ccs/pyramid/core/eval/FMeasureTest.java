package edu.neu.ccs.pyramid.core.eval;

public class FMeasureTest {
    public static void main(String[] args) {
        test1();
    }

    private static void test1(){
        double precision = 0.6;
        double recall = 0.7;
        System.out.println(FMeasure.f1(precision,recall));
    }

}