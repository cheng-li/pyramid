package edu.neu.ccs.pyramid.classification.naive_bayes;

import org.apache.commons.lang3.ArrayUtils;

import java.util.Arrays;

/**
 * Created by Rainicy on 10/6/14.
 *
 * Histogram ditribution, implementing Distribution.
 * And using Laplace Smoothing during the fit operation.
 *
 * @see edu.neu.ccs.pyramid.classification.naive_bayes.Distribution
 */
public class Histogram implements Distribution {


    /** numBins. */
    static public int numBins;
    /** Values for each bin. */
    protected double[] values;

    /** Minimum value in the histogram. */
    protected double min;
    /** Maximum value in the histogram. */
    protected double max;
    /** Each step */
    protected double step;
    /** default log density by given variable is 0. */
    private double defaultZeroLogProb;


    /** Default constructor, set numBins equals 2. */
    public Histogram() {
        this(2);
    }

    /**  Constructor by given numBins. */
    public Histogram(int numBins) {
        this.numBins = numBins;
        values = new double[numBins];
    }

    /** Constructor by given numBins and variables. */
    public Histogram(int numBins, double[] nonzeroVars, int numPerClass) {
        this(numBins);
        fit(nonzeroVars, numPerClass);
    }

    /**
     * Use the Laplace smoothing.
     * (counts + 1) / (Total_counts + numBins.)
     *
     * @param nonzeroVars nonzero variables array.
     * @param numPerClass total number of variables including zeros.
     * @throws IllegalArgumentException
     */
    public void fit(double[] nonzeroVars, int numPerClass) throws IllegalArgumentException{

        double[] zeroVars = new double[numPerClass];
        double[] variables = ArrayUtils.addAll(nonzeroVars, zeroVars);


        this.min = Arrays.stream(variables).min().getAsDouble();
        this.max = Arrays.stream(variables).max().getAsDouble();

        if (this.min > this.max) {
            throw new IllegalArgumentException("Minimum value" +
                    " should be smaller than Maximum");
        }

        this.step = (max-min) / this.numBins;

        // value for each bin
        int[] counts = new int[this.numBins];
        for (int i=0; i<variables.length; i++) {
            int binIndex = getIndexOfBins(variables[i]);
            counts[binIndex] += 1;
        }

        // Smoothing here.
        for (int i=0; i<this.numBins; i++) {
            this.values[i] = ((double)counts[i]+1)/(variables.length + this.numBins);
        }

        this.defaultZeroLogProb = logProbability(0.0);
    }


    /** By given a variable, find the index bin for this variable. */
    private int getIndexOfBins(double x) {

        double distance = x - this.min;
        int index = (int) (distance / this.step);

        if (index < 0) {
            return 0;
        }
        else if (index > this.numBins-1) {
            return this.numBins-1;
        }

        return index;
    }

    @Override
    public double probability(double x) {
        return values[getIndexOfBins(x)];
    }

    @Override
    public double logProbability(double x) throws IllegalArgumentException {
        if (x == 0.0) {
            return this.defaultZeroLogProb;
        }
        return Math.log(values[getIndexOfBins(x)]);
    }


    public String toString() {
        String str;
        str = "Total numBins: " + this.numBins + "\n";
        for (int i=0; i<this.numBins; i++) {
            str += "(" + i + ") \t" + this.values[i] + "\n";
        }
        return str;
    }

}
