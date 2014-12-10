package edu.neu.ccs.pyramid.optimization;

import org.apache.mahout.math.Vector;

/**
 * Created by chengli on 12/7/14.
 */
public class GradientDescent {
    private Optimizable.ByGradientValue function;
    private BackTrackingLineSearcher lineSearcher;


    public GradientDescent(Optimizable.ByGradientValue function,
                           double initialStepLength) {
        this.function = function;
        this.lineSearcher = new BackTrackingLineSearcher(function);
        lineSearcher.setInitialStepLength(initialStepLength);
    }

    public void optimize(int numIterations){



    }

    public void update(){
        Vector parameters = function.getParameters();
        Vector gradient = function.getGradient();
        Vector direction = gradient.times(-1);
        double stepLength = lineSearcher.findStepLength(direction);
        System.out.println("stepLength="+stepLength);
        Vector updatedParams = parameters.plus(direction.times(stepLength));
        parameters.assign(updatedParams);
        function.refresh();
    }
}
