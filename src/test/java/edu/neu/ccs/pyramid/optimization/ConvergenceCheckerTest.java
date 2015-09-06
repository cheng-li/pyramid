package edu.neu.ccs.pyramid.optimization;

import static org.junit.Assert.*;

public class ConvergenceCheckerTest {
    public static void main(String[] args) {
        double[] values = {1,2,-1,4,5,5.1,5.11,5.11,3,3.1,3.11,3.09,3.08,3.07,3.08};
        ConvergenceChecker checker = new ConvergenceChecker();
        for (double value: values){
            checker.add(value);
            System.out.println("iteration " + checker.getNumIterations());
            System.out.println("his="+checker.getHistory());
            System.out.println("min="+checker.getMinValue());
            System.out.println("max="+checker.getMaxValue());
            System.out.println("stable="+checker.getStableIterations());
            System.out.println("converge="+checker.isConverged());
        }
    }


}