package edu.neu.ccs.pyramid.optimization;

import org.apache.mahout.math.Vector;

/**
 * minimize the function along the line
 * Numerical Optimization, Second Edition, Jorge Nocedal Stephen J. Wright
 * algorithm: page 37
 * default parameters: page 142
 * Created by chengli on 12/9/14.
 */
public class BackTrackingLineSearcher {
    private Optimizable.ByGradientValue function;
    private double initialStepLength = 1;
    private double shrinkage = 0.5;
    private double c = 1e-4;

    public BackTrackingLineSearcher(Optimizable.ByGradientValue function) {
        this.function = function;
    }

    public double findStepLength(Vector searchDirection){
        double stepLength = initialStepLength;
        double value = function.getValue(function.getParameters());
        Vector gradient = function.getGradient();
        double product = gradient.dot(searchDirection);
        while(true){
            Vector step = searchDirection.times(stepLength);
            Vector target = function.getParameters().plus(step);
            double targetValue = function.getValue(target);
            if (targetValue <= value + c*stepLength*product){
                break;
            }
            stepLength *= shrinkage;
        }
        return stepLength;
    }



    public void setInitialStepLength(double initialStepLength) {
        this.initialStepLength = initialStepLength;
    }

    public void setShrinkage(double shrinkage) {
        this.shrinkage = shrinkage;
    }

    public void setC(double c) {
        this.c = c;
    }
}
