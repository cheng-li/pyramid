package edu.neu.ccs.pyramid.classification.naive_bayes;

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
                Math.log(standardDeviation) + 0.5 * Math.log(2 * Math.PI);
    }

    /** Gaussian constructor by given variables. */
    public Gaussian(double[] nonzeroVars, int numPerClass) {
        fit(nonzeroVars, numPerClass);
    }

    /**
     * Using following method to calculate the mean and variance.
     * refer: http://www.cs.berkeley.edu/~mhoemmen/cs194/Tutorials/variance.pdf
     * @param nonzeroVars
     * @param numPerClass
     * @throws IllegalArgumentException
     */
    public void fit(double[] nonzeroVars, int numPerClass) throws IllegalArgumentException {
        if (nonzeroVars.length == 0) {
            this.mean = 0;
            this.standardDeviation = 0;
            return;
        }

        double Mk = nonzeroVars[0];
        double Qk = 0;

        int count = 1;
        for(; count<nonzeroVars.length; count++) {
            double diff = nonzeroVars[count] - Mk;
            int k = count+1;
            Mk += diff / k;
            Qk += (count * diff * diff) / k;
        }
        // continue with zero variables
        // Mn = [(k-1)/(k+n-1)] * Mk
        // Qn = Qk + (Mk*Mk) * [(k-1)*n/(k+n-1)]
        int n = numPerClass - nonzeroVars.length;
        Qk += Mk * Mk * count * n / (count + n);
        Mk = Mk * count / (count + n);

        this.mean = Mk;
        this.standardDeviation = Math.sqrt(Qk / numPerClass);
        if (standardDeviation != 0.0) {
            this.logStandardVariancePlusHalfLog2Pi =
                    Math.log(standardDeviation) + 0.5 * Math.log(2 * Math.PI);
        }

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

        if (this.standardDeviation == 0.0) {
            return 0;
        }
        double x0 = x - mean;
        double x1 = x0 / standardDeviation;
        return -0.5 * x1 * x1 - logStandardVariancePlusHalfLog2Pi;
    }
}
