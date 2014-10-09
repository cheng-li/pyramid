package edu.neu.ccs.pyramid.classification.naive_bayes;

/**
 * Created by Rainicy on 10/6/14.
 *
 * Interface for probability model.
 */
public interface Probability {

    /**
     * The default threshold for rounding errors, to check
     * if probabilities sum to 1.
     */
    final double THRESHOLD = 1e-7;

    /**
     * Check if the probabilities is between [0-1],
     * and sum to 1.
     *
     * @return true, if valid, false otherwise.
     */
    boolean isValid() throws IllegalAccessException;
}
