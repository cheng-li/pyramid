package edu.neu.ccs.pyramid.classification.naive_bayes;

import org.apache.commons.lang3.ArrayUtils;
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
    /** default log density by given variable is 0. */
    private double defaultZeroLogProb;

    /** Defualt constructor with standard normal distribution. */
    public Gaussian() {
        this(0, 1);
    }

    /** Gaussian constructor by given mean and variance. */
    public Gaussian(double mean, double standardDeviation) {
        this.mean = mean;
        this.standardDeviation = standardDeviation;
        this.logStandardVariancePlusHalfLog2Pi =
                Math.log(standardDeviation) +
                        0.5 * FastMath.log(2 * FastMath.PI);
    }

    /** Gaussian constructor by given variables. */
    public Gaussian(double[] nonzeroVars, int numPerClass) {
        fit(nonzeroVars, numPerClass);
    }

    @Override
    public void fit(double[] nonzeroVars, int numPerClass) throws IllegalArgumentException {

        double mean = StatUtils.sum(nonzeroVars) / numPerClass;
        double[] zeroVars = new double[numPerClass];
        double[] variables = ArrayUtils.addAll(nonzeroVars, zeroVars);

        this.mean = mean;
        this.standardDeviation = Math.sqrt(StatUtils.variance(variables));
        this.defaultZeroLogProb = logProbability(0);
    }

    @Override
    public double probability(double x) throws IllegalArgumentException {
        return FastMath.exp(logProbability(x));
    }

    /**
     * if the standard deviation is 0, then return 0.
     * else return value of:
     * -log(sigma)-log(\sqrt{2\pi}) - (x - mean)^2 / 2(sigma)^2
     * @param x
     * @return
     * @throws IllegalArgumentException
     */
    @Override
    public double logProbability(double x) throws IllegalArgumentException {

        if (x == 0.0) {
            return this.defaultZeroLogProb;
        }

        if (this.standardDeviation == 0.0) {
            return 0;
        }
        double x0 = x - mean;
        double x1 = x0 / standardDeviation;
        return -0.5 * x1 * x1 - logStandardVariancePlusHalfLog2Pi;
    }
}
