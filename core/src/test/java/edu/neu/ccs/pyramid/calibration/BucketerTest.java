package edu.neu.ccs.pyramid.calibration;


public class BucketerTest {
    public static void main(String[] args) {
        test1();
    }


    private static void test1(){
        double[] x = {0.2,0.1,0.3,0.6,0.4,0.5,0.8,0.7,0.9};
        double[] y = {2,1,3,6,4,5,8,7,9};
        System.out.println(Bucketer.groupWithEqualSize(x,y,3));
        System.out.println(Bucketer.groupWithEqualSize(x,y,4));
        System.out.println(Bucketer.groupWithEqualSize(x,y,2));
    }

}