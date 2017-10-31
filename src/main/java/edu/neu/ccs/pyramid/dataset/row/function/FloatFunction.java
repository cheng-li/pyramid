package edu.neu.ccs.pyramid.dataset.row.function;

/**
 * Created by Rainicy on 10/30/17.
 * Reimplement the DoubleFunction from org.apache.mahout.math.function by
 *
 * 1) changing the Double to Float
 */
public abstract class FloatFunction {

    /**
     * Apply the function to the argument and return the result
     *
     * @param x double for the argument
     * @return the result of applying the function
     */
    public abstract float apply(float x);
}
