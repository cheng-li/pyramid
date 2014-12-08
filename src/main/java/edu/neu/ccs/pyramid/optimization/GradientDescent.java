package edu.neu.ccs.pyramid.optimization;

import org.apache.mahout.math.Vector;

/**
 * Created by chengli on 12/7/14.
 */
public class GradientDescent {
    private Optimizable.ByGradient function;
    double learningRate;

    public GradientDescent(Optimizable.ByGradient function, double learningRate) {
        this.function = function;
        this.learningRate = learningRate;
    }

    public void optimize(int numIterations){



    }

    public void update(){
        Vector parameters = function.getParameters();
        Vector gradient = function.getGradient();
        Vector updatedParams = parameters.minus(gradient.times(learningRate));
        parameters.assign(updatedParams);

    }
}
