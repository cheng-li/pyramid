package edu.neu.ccs.pyramid.core.util;

/**
 * Created by chengli on 7/19/16.
 */
public class ArgMax {
    public static int argMax(double[] array){
        int res = 0;
        double max = Double.NEGATIVE_INFINITY;
        for (int i=0;i<array.length;i++){
            double v = array[i];
            if (v> max){
                max = v;
                res = i;
            }
        }
        return res;
    }
}
