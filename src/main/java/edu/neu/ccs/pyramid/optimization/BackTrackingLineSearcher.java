package edu.neu.ccs.pyramid.optimization;

import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.apache.mahout.math.Vector;

/**
 * minimize the function along the line
 * Numerical Optimization, Second Edition, Jorge Nocedal Stephen J. Wright
 * algorithm: page 37
 * default parameters: page 142
 * Created by chengli on 12/9/14.
 */
public class BackTrackingLineSearcher {
    private static final Logger logger = LogManager.getLogger();
    private Optimizable.ByGradientValue function;
    private double initialStepLength = 1;
    private double shrinkage = 0.5;
    private double c = 1e-4;

    public BackTrackingLineSearcher(Optimizable.ByGradientValue function) {
        this.function = function;
    }

    public double findStepLength(Vector searchDirection){
        if (logger.isDebugEnabled()){
            logger.debug("start line search");
        }
        double stepLength = initialStepLength;
        double value = function.getValue();
        Vector gradient = function.getGradient();
        double product = gradient.dot(searchDirection);
        while(true){
            Vector step = searchDirection.times(stepLength);
            Vector target = function.getParameters().plus(step);

            double targetValue = function.newInstance(target).getValue();
            if (targetValue <= value + c*stepLength*product){
                break;
            }
            stepLength *= shrinkage;
        }
        if (logger.isDebugEnabled()){
            logger.debug("line search done. Step length = "+stepLength);
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
