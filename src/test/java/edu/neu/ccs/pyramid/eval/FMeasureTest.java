package edu.neu.ccs.pyramid.eval;

import static org.junit.Assert.*;

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