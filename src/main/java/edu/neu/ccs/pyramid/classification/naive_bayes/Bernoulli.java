package edu.neu.ccs.pyramid.classification.naive_bayes;

import org.apache.commons.math3.util.FastMath;

/**
 * Created by Rainicy on 10/10/14.
 */
public class Bernoulli implements Distribution {

    /** Phi is the ratio of non-zero to
     *  total counts. And using smoothing. */
    private double phi;


    /** Default constructor */
    public Bernoulli(double phi) {
        this.phi = phi;
    }

    /** Constructor by given variables */
    public Bernoulli(double[] variables) {
        fit(variables);
    }

    @Override
    public void fit(double[] variables) throws IllegalArgumentException {
        if (variables.length <= 0) {
            throw new IllegalArgumentException("Given variables should " +
                    "be more than 0.");
        }

        int countsNonZeros = 0;
        for (int i=0; i<variables.length; i++) {
            if (variables[i] != 0.0) {
                countsNonZeros++;
            }
        }
        // smoothing
        this.phi = ((double) countsNonZeros + 1) / (variables.length + 2);
    }

    @Override
    public double probability(double x) throws IllegalArgumentException {
        if (x != 0.0) {
            return getPhi();
        }
        return 1 - getPhi();
    }

    @Override
    public double logProbability(double x) throws IllegalArgumentException {
        return FastMath.log(probability(x));
    }

    @Override
    public double cumulativeProbability(double x) {
        if (x == 0) {
            return 1 - getPhi();
        }
        return 1;
    }

    @Override
    public double getMean() throws IllegalAccessException {
        return getPhi();
    }

    @Override
    public double getVariance() throws IllegalAccessException {
        return getPhi() * (1 - getPhi());
    }

    @Override
    public boolean isValid() throws IllegalAccessException {
        if ((getPhi() < 0) || (getPhi()>1)) {
            return false;
        }
        return true;
    }

    public double getPhi() {
        return this.phi;
    }
}
