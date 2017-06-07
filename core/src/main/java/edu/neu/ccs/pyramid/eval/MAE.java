package edu.neu.ccs.pyramid.eval;

/**
 * mean absolute error
 * Created by chengli on 3/22/17.
 */
public class MAE {
    public static double mae(double[] labels, double[] predictions){
        if (labels.length != predictions.length){
            throw new IllegalArgumentException("dimensions don't match");
        }
        double error = 0;
        for (int i=0;i<labels.length;i++){
            error += Math.abs(labels[i]-predictions[i]);
        }
        return error/labels.length;
    }
}
