package edu.neu.ccs.pyramid.eval;

import java.util.Arrays;

/**
 * relative absolute error
 * Created by chengli on 3/22/17.
 */
public class RAE {
    public static double rae(double[] labels, double[] predictions){
        double ave = Arrays.stream(labels).average().getAsDouble();
        double numerator = 0;
        double denominator = 0;

        for (int i=0;i<labels.length;i++){
            numerator += Math.abs(labels[i]-predictions[i]);
            denominator += Math.abs(labels[i]-ave);
        }
        return numerator/denominator;
    }
}
