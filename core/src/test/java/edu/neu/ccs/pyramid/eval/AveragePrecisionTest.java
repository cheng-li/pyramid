package edu.neu.ccs.pyramid.eval;



/**
 * Created by chengli on 12/27/16.
 */
public class AveragePrecisionTest {
    public static void main(String[] args) {
        test3();
    }

    private static void test1(){
        boolean[] re = {true, false, true, false, true};
        System.out.println(AveragePrecision.averagePrecision(re));
    }

    private static void test2(){
        boolean[] re = {true, false, true, false, true, false};
        System.out.println(AveragePrecision.averagePrecision(re));
    }


    private static void test3(){
        int[] labels = {1,0,1,0,1};
        double[] scores = {0.1,0.2, 0.3, 0.4, 0.5};

        System.out.println(AveragePrecision.averagePrecision(labels, scores));
    }

}