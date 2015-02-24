package edu.neu.ccs.pyramid.util;

import java.util.ArrayList;
import java.util.List;
import java.util.stream.Collectors;

/**
 * Created by chengli on 2/20/15.
 */
public class Grid {

    /**
     * uniform grid
     * from small to big
     * @param min
     * @param max
     * @param num
     * @return
     */
    public static List<Double> uniform(double min, double max, int num){
        if (min> max ||num<2){
            throw new IllegalArgumentException("min> max ||num<2");
        }

        double interval = (max-min)/(num-1);
        List<Double> values = new ArrayList<>();
        for (int i=0;i<num;i++){
            values.add(min+i*interval);
        }
        return values;
    }

    /**
     * log uniform grid
     * from small to big
     * @param min
     * @param max
     * @param num
     * @return
     */
    public static List<Double> logUniform(double min, double max, int num){
        if (min> max ||num<2 ||min<0){
            throw new IllegalArgumentException("min> max ||num<2");
        }

        return uniform(Math.log(min),Math.log(max),num).stream().map(Math::exp)
                .collect(Collectors.toList());
    }


}
