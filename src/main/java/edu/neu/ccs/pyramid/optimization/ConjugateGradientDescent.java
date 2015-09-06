package edu.neu.ccs.pyramid.optimization;

import org.apache.mahout.math.Vector;

/**
 * Numerical Optimization, Second Edition, Jorge Nocedal Stephen J. Wright
 * algorithm 5.4
 * Created by chengli on 12/9/14.
 */
public class ConjugateGradientDescent extends GradientValueOptimizer implements Optimizer{
    private BackTrackingLineSearcher lineSearcher;
    private Vector oldP;
    private Vector oldGradient;


    public ConjugateGradientDescent(Optimizable.ByGradientValue function) {
        super(function);
        this.lineSearcher = new BackTrackingLineSearcher(function);
        this.oldGradient = function.getGradient();
        this.oldP = oldGradient.times(-1);
    }

    public void iterate(){
        Vector direction = this.oldP;
        lineSearcher.moveAlongDirection(direction);
        Vector newGradient = function.getGradient();
        double beta = newGradient.dot(newGradient)/oldGradient.dot(oldGradient);
        Vector newP = oldP.times(beta).minus(newGradient);
        oldP = newP;
        oldGradient = newGradient;
    }

    public BackTrackingLineSearcher getLineSearcher() {
        return lineSearcher;
    }
}
