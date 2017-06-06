package edu.neu.ccs.pyramid.core.eval;

public class PrecisionTest {
    public static void main(String[] args) {
        test1();
    }

    private static void test1(){
        int[] predictions = {0,0,1,1,1};
        int[] labels = {0,0,0,1,1};
        System.out.println(Precision.precision(labels,predictions,1));
    }

}