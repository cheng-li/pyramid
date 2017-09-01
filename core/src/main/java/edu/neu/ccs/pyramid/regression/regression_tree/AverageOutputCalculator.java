package edu.neu.ccs.pyramid.regression.regression_tree;

/**
 * default output of regression tree
 * Created by chengli on 8/5/14.
 */
public class AverageOutputCalculator implements LeafOutputCalculator{


    //todo: parallelize?
    @Override
    public double getLeafOutput(double[] probabilities, double[] labels) {
        double sum = 0;
        double count = 0;
        for (int i=0;i<probabilities.length;i++){
            sum += labels[i]*probabilities[i];
            count += probabilities[i];
        }
        return sum/count;
    }
}
