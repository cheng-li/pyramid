package edu.neu.ccs.pyramid.classification.naive_bayes;

/**
 * Created by Rainicy on 10/6/14.
 *
 * This is an interface for probability distribution.
 *
 * It can be extended by different distribution.
 */
public interface Distribution{

    /**
     * Given by batch of nonzero variables, fits its distribution.
     */
    public void fit(double[] nonzeroVars, int numPerClass)
            throws IllegalArgumentException;

    /**
     * Given a variable, calculate its probability.
     */
    public double probability(double x) throws IllegalArgumentException;

    /**
     * GIven a variable, calculate its log probability.
     */
    public double logProbability(double x) throws  IllegalArgumentException;

}
