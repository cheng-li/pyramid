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
    private double standardVariance;
    /** The value of {log(std) + 0.5*log(2*pi) } */
    private double logStandardVariancePlusHalfLog2Pi;

    /** Defualt constructor with standard normal distribution. */
    public Gaussian() {
        this(0, 1);
    }

    /** Gaussian constructor by given mean and variance. */
    public Gaussian(double mean, double standardVariance) {
        this.mean = mean;
        this.standardVariance = standardVariance;
        this.logStandardVariancePlusHalfLog2Pi =
                FastMath.log(standardVariance) +
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
        setStandardVariance(FastMath.sqrt(StatUtils.variance(variables)));
    }

    @Override
    public double probability(double x) throws IllegalArgumentException {
        return FastMath.exp(logProbability(x));
    }

    @Override
    public double logProbability(double x) throws IllegalArgumentException {
        double x0 = x - mean;
        double x1 = x0 / standardVariance;
        return -0.5 * x1 * x1 - logStandardVariancePlusHalfLog2Pi;
    }

    @Override
    public double cumulativeProbability(double x) {
        double dev = x - mean;
        if (FastMath.abs(dev) > 40 * standardVariance) {
            return dev < 0 ? 0.0d : 1.0d;
        }
        return 0.5 * (1 + Erf.erf(dev / (standardVariance * FastMath.sqrt(2.0))));
    }

    @Override
    public double getMean() throws IllegalAccessException {
        return this.mean;
    }

    @Override
    public double getVariance() throws IllegalAccessException {
        return this.standardVariance * this.standardVariance;
    }

    @Override
    public boolean isValid() throws IllegalAccessException {
        throw new IllegalAccessException("Cannot support by " +
                "Gaussian Distribution.");
    }


    private void setStandardVariance(double standardVariance) {
        this.standardVariance = standardVariance;
    }

    private void setMean(double mean) {
        this.mean = mean;
    }
}
