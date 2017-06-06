package edu.neu.ccs.pyramid.core.util;

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
        if (min> max ||num<1){
            throw new IllegalArgumentException("min> max ||num<1");
        }
        List<Double> values = new ArrayList<>();
        if (num==1){
            if (min!=max){
                throw new IllegalArgumentException("num = 1 but min!=max");
            }
            values.add(min);
            return values;
        }

        double interval = (max-min)/(num-1);

        for (int i=0;i<num;i++){
            values.add(min+i*interval);
        }
        return values;
    }

    public static List<Double> uniformDecreasing(double min, double max, int num){
        List<Double> list = uniform(min,max,num);
        List<Double> des = new ArrayList<>();
        for (int i=0;i<list.size();i++){
            des.add(list.get(list.size()-i-1));
        }
        return des;
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
        if (min> max ||num<1 ||min<0){
            throw new IllegalArgumentException("min> max ||num<1");
        }

        return uniform(Math.log(min),Math.log(max),num).stream().map(Math::exp)
                .collect(Collectors.toList());
    }


}
