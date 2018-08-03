package edu.neu.ccs.pyramid.regression.regression_tree;

import java.util.stream.IntStream;

/**
 * default output of regression tree
 * Created by chengli on 8/5/14.
 */
public class AverageOutputCalculator implements LeafOutputCalculator{
    private boolean parallel;


    //todo: parallelize?
    @Override
    public double getLeafOutput(double[] probabilities, double[] labels) {
        double sum = 0;
        double count = 0;
        IntStream intStream = IntStream.range(0,probabilities.length);
        if (parallel){
            intStream = intStream.parallel();
        }
        IntStream intStream2 = IntStream.range(0,probabilities.length);
        if (parallel){
            intStream2 = intStream2.parallel();
        }
        sum = intStream.mapToDouble(i->labels[i]*probabilities[i]).sum();
        count = intStream2.mapToDouble(i->probabilities[i]).sum();
//        for (int i=0;i<probabilities.length;i++){
//            sum += labels[i]*probabilities[i];
//            count += probabilities[i];
//        }
        return sum/count;
    }

    @Override
    public void setParallel(boolean parallel) {

    }
}
