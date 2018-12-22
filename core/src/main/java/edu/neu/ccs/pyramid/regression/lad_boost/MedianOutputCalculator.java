package edu.neu.ccs.pyramid.regression.lad_boost;

import edu.neu.ccs.pyramid.regression.regression_tree.LeafOutputCalculator;
import edu.neu.ccs.pyramid.util.MathUtil;

import java.util.ArrayList;
import java.util.List;

/**
 * Created by chengli on 10/8/16.
 */
public class MedianOutputCalculator implements LeafOutputCalculator {
    @Override
    public double getLeafOutput(double[] probabilities, double[] labels) {
        // todo
        return MathUtil.weightedMedian(labels, probabilities);

//        List<Double> nonZeros = new ArrayList<>();
//        for (int i=0;i<probabilities.length;i++){
//            if (probabilities[i]!=0){
//                nonZeros.add(labels[i]);
//            }
//        }
//
//        double[] a = nonZeros.stream().mapToDouble(b->b).toArray();
//        return MathUtil.median(a);
    }

    @Override
    public void setParallel(boolean parallel) {

    }
}
