package edu.neu.ccs.pyramid.eval;

import java.util.stream.IntStream;

/**
 * mean relative error
 * Created by chengli on 3/22/17.
 */
public class MRE {
    public static double mre(double[] labels, double[] predictions){
        return IntStream.range(0, labels.length).mapToDouble(i->Math.abs(labels[i]-predictions[i])/labels[i]).average().getAsDouble();
    }
}
