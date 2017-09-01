package edu.neu.ccs.pyramid.util;

import java.util.List;

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


    public static int argMax(List<Double> list){
        int res = 0;
        double max = Double.NEGATIVE_INFINITY;
        for (int i=0;i<list.size();i++){
            double v = list.get(i);
            if (v> max){
                max = v;
                res = i;
            }
        }
        return res;
    }
}
