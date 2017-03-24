package edu.neu.ccs.pyramid.optimization;


import org.apache.mahout.math.Vector;


/**
 * Created by chengli on 12/7/14.
 */
public class GradientDescent extends GradientValueOptimizer implements Optimizer.Iterative {
    private BackTrackingLineSearcher lineSearcher;

    public GradientDescent(Optimizable.ByGradientValue function) {
        super(function);
        this.lineSearcher = new BackTrackingLineSearcher(function);
    }

    public BackTrackingLineSearcher getLineSearcher() {
        return lineSearcher;
    }



    public void iterate(){
        Vector gradient = this.function.getGradient();
        Vector direction = gradient.times(-1);
        lineSearcher.moveAlongDirection(direction);
        terminator.add(function.getValue());
    }

}
