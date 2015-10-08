package edu.neu.ccs.pyramid.optimization;

import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

/**
 * Created by chengli on 9/6/15.
 */
public abstract class GradientValueOptimizer implements Optimizer{
    protected Terminator terminator;
    protected Optimizable.ByGradientValue function;

    public GradientValueOptimizer(Optimizable.ByGradientValue function) {
        this.terminator = new Terminator();
        this.terminator.setGoal(Terminator.Goal.MINIMIZE);
        this.function = function;
    }

    @Override
    public void optimize() {
        while(true){
            iterate();
            terminator.add(function.getValue());
            if (terminator.shouldTerminate()){
                break;
            }
        }
    }

    abstract protected void iterate();

    @Override
    public double getFinalObjective() {
        return this.terminator.getLastValue();
    }

    @Override
    public Terminator getTerminator() {
        return this.terminator;
    }
}
