package edu.neu.ccs.pyramid.optimization;

import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.apache.mahout.math.Vector;

import java.util.LinkedList;

/**
 * Created by chengli on 12/7/14.
 */
public class GradientDescent implements Optimizer{
    private static final Logger logger = LogManager.getLogger();
    private Optimizable.ByGradientValue function;
    private BackTrackingLineSearcher lineSearcher;
    private Terminator terminator;


    public GradientDescent(Optimizable.ByGradientValue function) {
        this.function = function;
        this.lineSearcher = new BackTrackingLineSearcher(function);
        this.terminator = new Terminator();
    }

    public BackTrackingLineSearcher getLineSearcher() {
        return lineSearcher;
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
        Vector gradient = function.getGradient();
        Vector direction = gradient.times(-1);
        lineSearcher.moveAlongDirection(direction);

    }

}
