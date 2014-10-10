package edu.neu.ccs.pyramid.classification.naive_bayes;

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
    public void fit(double[] variables)
            throws IllegalArgumentException;

    /**
     * Given a variable, calculate its probability.
     */
    public double probability(double x) throws IllegalArgumentException;

    /**
     * GIven a variable, calculate its log probability.
     */
    public double logProbability(double x) throws  IllegalArgumentException;

    /**
     * Given a varibale, return its cumulative probability.
     */
    public double cumulativeProbability(double x) throws IllegalAccessException;


    /**
     * Returns the mean value of this distribution.
     */
    public double getMean() throws IllegalAccessException;

    /**
     * Returns the variance of this distribution.
     */
    public double getVariance() throws IllegalAccessException;
}
