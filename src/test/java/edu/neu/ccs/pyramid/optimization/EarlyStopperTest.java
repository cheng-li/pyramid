package edu.neu.ccs.pyramid.optimization;

import junit.framework.TestCase;

/**
 * Created by chengli on 10/29/16.
 */
public class EarlyStopperTest {

    public static void main(String[] args) {
        test3();
    }


    private static void test1(){
        EarlyStopper earlyStopper = new EarlyStopper(EarlyStopper.Goal.MAXIMIZE, 2);
        double[] values = {0.5, 0.6, 0.5, 0.5, 0.7};
        earlyStopper.setMinimumIterations(2);
        for (int i=0;i<values.length;i++){
            earlyStopper.add(i, values[i]);
            System.out.println(earlyStopper.shouldStop());
            System.out.println(earlyStopper.getBestIteration());
        }
    }

    private static void test2(){
        EarlyStopper earlyStopper = new EarlyStopper(EarlyStopper.Goal.MAXIMIZE, 2);
        earlyStopper.setMinimumIterations(2);
        double[] values = {0.5, 0.6, 0.5, 0.7, 0.5, 0.4, 0.8};
        for (int i=0;i<values.length;i++){
            earlyStopper.add(i, values[i]);
            System.out.println(earlyStopper.shouldStop());
            System.out.println(earlyStopper.getBestIteration());
        }
        System.out.println(earlyStopper.history());
    }


    private static void test3(){
        EarlyStopper earlyStopper = new EarlyStopper(EarlyStopper.Goal.MINIMIZE, 2);
        double[] values = {0.5, 0.6, 0.4, 0.7, 0.5, 0.4, 0.8};
        earlyStopper.setMinimumIterations(2);
        for (int i=0;i<values.length;i++){
            earlyStopper.add(i, values[i]);
            System.out.println(earlyStopper.shouldStop());
            System.out.println(earlyStopper.getBestIteration());
        }
        System.out.println(earlyStopper.history());
    }

}