package edu.neu.ccs.pyramid.util;

import java.util.List;

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


    public static int argMin(List<Double> list){
        int res = 0;
        double min = Double.POSITIVE_INFINITY;
        for (int i=0;i<list.size();i++){
            double v = list.get(i);
            if (v< min){
                min = v;
                res = i;
            }
        }
        return res;
    }
}
