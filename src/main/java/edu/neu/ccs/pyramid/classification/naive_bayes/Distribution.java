package edu.neu.ccs.pyramid.classification.naive_bayes;

import edu.neu.ccs.pyramid.dataset.FeatureColumn;

/**
 * Created by Rainicy on 10/6/14.
 *
 * This is an interface for probability distribution.
 *
 * It can be extended by different distribution.
 */
public interface Distribution extends Probability{

    /**
     * Given by batch of data, fits its distribution.
     */
    public void fit(FeatureColumn featureColumn);

    /**
     * Given a variable, calculate its probability.
     */
    public Double probability(double x);

    /**
     * Given a varibale, return its cumulative probability.
     */
    public Double cumulativeProbability(double x);


    /**
     * Returns the mean value of this distribution.
     */
    public Double getMean();

    /**
     * Returns the variance of this distribution.
     */
    public Double getVariance();
}
