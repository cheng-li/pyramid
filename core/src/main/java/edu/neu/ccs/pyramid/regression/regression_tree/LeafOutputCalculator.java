package edu.neu.ccs.pyramid.regression.regression_tree;

/**
 * Created by chengli on 8/5/14.
 */
public interface LeafOutputCalculator {
    /**
     * by default, leaf output = average value
     * sometimes we want to set it to be something else
     * for example, using a newton step/ shrinkage rate
     * sometimes, we want to change the sign of output, depending on
     * whether we maximize or minimize something
     * @param probabilities: the probability of each data point at the leaf node
     * @return output of the leaf node
     */
    double getLeafOutput(double[] probabilities, double[] labels);
    void setParallel(boolean parallel);
}
