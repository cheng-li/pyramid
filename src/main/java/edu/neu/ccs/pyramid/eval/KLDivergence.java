package edu.neu.ccs.pyramid.eval;

import org.apache.commons.math3.util.FastMath;

/**
 * Created by chengli on 11/11/14.
 */
public class KLDivergence {
    public static double kl(double[] trueDistribution, double[] estimatedDistribution){
        double r = 0;
        for (int i=0;i<trueDistribution.length;i++){
            if (trueDistribution[i]==0){
                r += 0;
            } else {
                r += trueDistribution[i]* FastMath.log(2, trueDistribution[i] / estimatedDistribution[i]);
            }
        }
        return r;
    }
}
