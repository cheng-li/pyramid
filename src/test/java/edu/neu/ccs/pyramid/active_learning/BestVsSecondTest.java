package edu.neu.ccs.pyramid.active_learning;

import static org.junit.Assert.*;

public class BestVsSecondTest {
    public static void main(String[] args) {
        test1();
    }

    private static void test1(){
        double[] probs = {0.1,0.2,0.1,0.3,0.14,0.16};
        BestVsSecond bestVsSecond = new BestVsSecond(probs);
        System.out.println(bestVsSecond);
    }

}