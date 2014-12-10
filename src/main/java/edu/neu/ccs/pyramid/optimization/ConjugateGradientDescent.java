package edu.neu.ccs.pyramid.optimization;

import org.apache.mahout.math.Vector;

/**
 * Numerical Optimization, Second Edition, Jorge Nocedal Stephen J. Wright
 * algorithm 5.4
 * Created by chengli on 12/9/14.
 */
public class ConjugateGradientDescent {
    private Optimizable.ByGradientValue function;
    private BackTrackingLineSearcher lineSearcher;
    private Vector oldP;
    private Vector oldGradient;


    public ConjugateGradientDescent(Optimizable.ByGradientValue function,
                           double initialStepLength) {
        this.function = function;
        this.lineSearcher = new BackTrackingLineSearcher(function);
        lineSearcher.setInitialStepLength(initialStepLength);
        this.oldGradient = function.getGradient();
        this.oldP = oldGradient.times(-1);
    }

    public void optimize(int numIterations){



    }

    public void update(){
        Vector parameters = function.getParameters();
        Vector direction = this.oldP;
        double stepLength = lineSearcher.findStepLength(direction);
        System.out.println("stepLength="+stepLength);
        Vector updatedParams = parameters.plus(direction.times(stepLength));
        parameters.assign(updatedParams);
        function.refresh();
        Vector newGradient = function.getGradient();
        double beta = newGradient.dot(newGradient)/oldGradient.dot(oldGradient);
        Vector newP = oldP.times(beta).minus(newGradient);
        oldP = newP;
        oldGradient = newGradient;
    }
}
