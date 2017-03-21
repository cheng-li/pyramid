package edu.neu.ccs.pyramid.util;

/**
 * Created by chengli on 9/29/16.
 */
public class ArgMin {

    public static int argMin(double[] array){
        int res = 0;
        double min = Double.POSITIVE_INFINITY;
        for (int i=0;i<array.length;i++){
            double v = array[i];
            if (v< min){
                min = v;
                res = i;
            }
        }
        return res;
    }
}
