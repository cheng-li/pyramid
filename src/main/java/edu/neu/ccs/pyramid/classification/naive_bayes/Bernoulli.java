package edu.neu.ccs.pyramid.classification.naive_bayes;

import org.apache.commons.math3.util.FastMath;

/**
 * Created by Rainicy on 10/10/14.
 *
 * Bernoulli distribution.
 *
 * @see edu.neu.ccs.pyramid.classification.naive_bayes.Distribution
 */
public class Bernoulli implements Distribution {

    /** Phi is the ratio of non-zero to
     *  total counts. And using smoothing. */
    private double phi;
    private double logPosPhi;
    private double logNegPhi;


    /** Default constructor */
    public Bernoulli(double phi) {
        this.phi = phi;
        this.logPosPhi = FastMath.log(phi);
        this.logNegPhi = FastMath.log(1 - phi);
    }

    /** Constructor by given vector */
    public Bernoulli (double[] nonzeroVars, int numPerClass) {
        int countsNonZeros = nonzeroVars.length;

        // smoothing and fitting the parameters.
        this.phi = ((double) countsNonZeros + 1) / (numPerClass + 2);
        this.logPosPhi = Math.log(phi);
        this.logNegPhi = Math.log(1 - phi);
    }

    @Override
    public double logProbability(double x) throws IllegalArgumentException {
        if (x > 0.0) {
            return logPosPhi;
        }
        return logNegPhi;
    }
}
