package edu.neu.ccs.pyramid.optimization;

import org.apache.mahout.math.Vector;

/**
 * Numerical Optimization, Second Edition, Jorge Nocedal Stephen J. Wright
 * algorithm 5.4
 * Created by chengli on 12/9/14.
 */
public class ConjugateGradientDescent implements Optimizer{
    private Optimizable.ByGradientValue function;
    private BackTrackingLineSearcher lineSearcher;
    private Vector oldP;
    private Vector oldGradient;
    private Terminator terminator;


    public ConjugateGradientDescent(Optimizable.ByGradientValue function,
                           double initialStepLength) {
        this.function = function;
        this.lineSearcher = new BackTrackingLineSearcher(function);
        lineSearcher.setInitialStepLength(initialStepLength);
        this.terminator = new Terminator();
        this.oldGradient = function.getGradient();
        this.oldP = oldGradient.times(-1);
    }



    public void optimize(){
        while(true){
            iterate();
            terminator.add(function.getValue());
            if (terminator.shouldTerminate()){
                break;
            }
        }
    }

    @Override
    public double getFinalObjective() {
        return terminator.getLastValue();
    }

    @Override
    public Terminator getTerminator() {
        return terminator;
    }

    public void iterate(){
        Vector parameters = function.getParameters();
        Vector direction = this.oldP;
        lineSearcher.moveAlongDirection(direction);
        Vector newGradient = function.getGradient();
        double beta = newGradient.dot(newGradient)/oldGradient.dot(oldGradient);
        Vector newP = oldP.times(beta).minus(newGradient);
        oldP = newP;
        oldGradient = newGradient;
    }
}
