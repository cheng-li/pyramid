package edu.neu.ccs.pyramid.util;

/**
 * Created by chengli on 10/23/15.
 */
public class Sigmoid {
    /**
     * 1/(1+exp(-x))
     * no way to make it accurate
     * @param x
     * @return
     */
    public static double sigmoid(double x){
        return 1.0/(1+Math.exp(-x));
    }

    /**
     * high precision
     * log of sigmoid
     * @param x
     * @return
     */
    public static double logSidmoid(double x){
        double[] arr = {x,0};
        return x - MathUtil.logSumExp(arr);
    }
}
