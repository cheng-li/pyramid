package edu.neu.ccs.pyramid.core.optimization;

import org.apache.mahout.math.DenseVector;
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
        // we need to make a copy of the gradient; should not use pointer
        this.oldGradient = new DenseVector(function.getGradient());
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
        terminator.add(function.getValue());
    }

    public BackTrackingLineSearcher getLineSearcher() {
        return lineSearcher;
    }
}
