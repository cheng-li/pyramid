package edu.neu.ccs.pyramid.eval;

import edu.neu.ccs.pyramid.util.MathUtil;

/**
 * Created by chengli on 10/8/16.
 */
public class Pearson {

    public static double pearson(double[] x, double[] y){
        double xave = MathUtil.arraySum(x)/x.length;
        double yave = MathUtil.arraySum(y)/y.length;
        double prod = 0;
        double xvar = 0;
        double yvar = 0;
        for (int i=0;i<x.length;i++){
            prod += (x[i]-xave)*(y[i]-yave);
            xvar += Math.pow(x[i]-xave,2);
            yvar += Math.pow(y[i]-yave,2);
        }

        return prod/(Math.sqrt(xvar*yvar));
    }
}
