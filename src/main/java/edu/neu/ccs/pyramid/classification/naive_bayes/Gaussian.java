package edu.neu.ccs.pyramid.classification.naive_bayes;

import org.apache.commons.math3.special.Erf;
import org.apache.commons.math3.stat.StatUtils;
import org.apache.commons.math3.util.FastMath;

/**
 * Created by Rainicy on 10/8/14.
 *
 * Gaussian distribution implements Distribution.
 *
 * @see edu.neu.ccs.pyramid.classification.naive_bayes.Distribution
 *
 * @refrence org.apache.commons.math3.disribution.NormalDistribution.
 */
public class Gaussian implements Distribution {

    /** Mean */
    private double mean;
    /** Variance */
    private double standardDeviation;
    /** The value of {log(std) + 0.5*log(2*pi) } */
    private double logStandardVariancePlusHalfLog2Pi;

    /** Defualt constructor with standard normal distribution. */
    public Gaussian() {
        this(0, 1);
    }

    /** Gaussian constructor by given mean and variance. */
    public Gaussian(double mean, double standardDeviation) {
        this.mean = mean;
        this.standardDeviation = standardDeviation;
        this.logStandardVariancePlusHalfLog2Pi =
                FastMath.log(standardDeviation) +
                        0.5 * FastMath.log(2 * FastMath.PI);
    }

    /** Gaussian constructor by given variables. */
    public Gaussian(double[] variables) {
        fit(variables);
    }


    @Override
    public void fit(double[] variables) throws IllegalArgumentException {
        if (variables.length <= 0) {
            throw new IllegalArgumentException("Given variables should " +
                    "be more than 0.");
        }
        setMean(StatUtils.mean(variables));
        setStandardDeviation(FastMath.sqrt(StatUtils.variance(variables)));
    }

    @Override
    public double probability(double x) throws IllegalArgumentException {
        return FastMath.exp(logProbability(x));
    }

    @Override
    public double logProbability(double x) throws IllegalArgumentException {
        double x0 = x - mean;
        double x1 = x0 / standardDeviation;
        return -0.5 * x1 * x1 - logStandardVariancePlusHalfLog2Pi;
    }

    @Override
    public double cumulativeProbability(double x) {
        double dev = x - mean;
        if (FastMath.abs(dev) > 40 * standardDeviation) {
            return dev < 0 ? 0.0d : 1.0d;
        }
        return 0.5 * (1 + Erf.erf(dev / (standardDeviation * FastMath.sqrt(2.0))));
    }

    @Override
    public double getMean() throws IllegalAccessException {
        return this.mean;
    }

    @Override
    public double getVariance() throws IllegalAccessException {
        return this.standardDeviation * this.standardDeviation;
    }

    @Override
    public boolean isValid() throws IllegalAccessException {
        throw new IllegalAccessException("Cannot support by " +
                "Gaussian Distribution.");
    }


    private void setStandardDeviation(double standardDeviation) {
        this.standardDeviation = standardDeviation;
    }

    private void setMean(double mean) {
        this.mean = mean;
    }
}
