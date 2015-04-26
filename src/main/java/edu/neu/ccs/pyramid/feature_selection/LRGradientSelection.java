package edu.neu.ccs.pyramid.feature_selection;

/**
 * Created by chengli on 4/25/15.
 */
public class LRGradientSelection {

    /**
     *
     * @param distribution ngram class distribution
     * @param probs probability produced by classifiers
     * @param classIndex
     * @return
     */
    public static double utility(NgramClassDistribution distribution, double[] probs, int classIndex){
        double actual = distribution.getClassCount(classIndex);
        double expected = ((double)distribution.getTotalCount())*probs[classIndex];
        return actual - expected;
    }
}
