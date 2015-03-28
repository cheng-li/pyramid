package edu.neu.ccs.pyramid.classification.naive_bayes;

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
    protected int numBins;
    /** Values for each bin. */
    protected double[] values;

    /** Minimum value in the histogram. */
    protected double min;
    /** Maximum value in the histogram. */
    protected double max;
    /** Each step */
    protected double step;


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
    public Histogram(int numBins, double[] variables) {
        this(numBins);
        fit(variables);
    }

    /**
     * Use the Laplace smoothing.
     * (counts + 1) / (Total_counts + numBins.)
     *
     * @param variables double array.
     * @throws IllegalArgumentException
     */
    @Override
    public void fit(double[] variables) throws IllegalArgumentException{

        setMin(Arrays.stream(variables).min().getAsDouble());
        setMax(Arrays.stream(variables).max().getAsDouble());

        if (getMin() > getMax()) {
            throw new IllegalArgumentException("Minimum value" +
                    " should be smaller than Maximum");
        }

        setStep( (double)(max-min) / getBins() );
        double start = getMin();

        // value for each bin
        int[] counts = new int[getBins()];
        int totalNumVariables = variables.length;
        for (int i=0; i<totalNumVariables; i++) {
            int binIndex = getIndexOfBins(variables[i]);
            counts[binIndex] += 1;
        }


        // Smoothing here.
        for (int i=0; i<getBins(); i++) {
            double value = ((double)counts[i]+1)/(totalNumVariables + getBins());
            setValue(value, i);
        }
    }

    /** By given a variable, find the index bin for this variable. */
    private int getIndexOfBins(double x) {

        double distance = x - getMin();
        int index = (int) (distance / getStep());

        if (index < 0) {
            return 0;
        }
        else if (index > getBins()-1) {
            return getBins()-1;
        }

        return index;
    }

    @Override
    public double probability(double x) {
        return getValue(getIndexOfBins(x));
    }

    @Override
    public double logProbability(double x) throws IllegalArgumentException {
        return Math.log(probability(x));
    }

    @Override
    public double cumulativeProbability(double x) {
        int index = getIndexOfBins(x);
        double cum = 0;
        for (int i=0; i<=index; i++) {
            cum += getValue(i);
        }
        return cum;
    }

    /**
     Not Support in Histogram.
      */
    @Override
    public double getMean() throws IllegalAccessException {
        throw new IllegalAccessException("Histogram does not support" +
                "getMean() operation.");
    }

    /**
    Not Support in Histogram.
     */
    @Override
    public double getVariance() throws IllegalAccessException {
        throw new IllegalAccessException("Histogram does not support" +
                "getVariance() operation.");
    }

    @Override
    public boolean isValid() {
        double sum = 0;
        for (int i=0; i<getBins(); i++) {
            double prob = getValue(i);
            if (prob < 0 || prob > 1) {
                return false;
            }
            sum += prob;
        }
        if (Math.abs(1-sum) > THRESHOLD) {
            return false;
        }
        return true;
    }

    public String toString() {
        String str;
        str = "Total numBins: " + getBins() + "\n";
        for (int i=0; i<getBins(); i++) {
            str += "(" + i + ") \t" + getValue(i) + "\n";
        }
        return str;
    }

    public int getBins() {
        return numBins;
    }

    private void setBins(int numBins) {
        this.numBins = numBins;
    }

    private double getValue(int index) {
        return values[index];
    }

    private void setValue(double value, int index) {
        this.values[index] = value;
    }

    private double getMax() {
        return max;
    }

    private void setMax(double max) {
        this.max = max;
    }

    private double getMin() {
        return min;
    }

    private void setMin(double min) {
        this.min = min;
    }

    private double getStep() {
        return step;
    }

    private void setStep(double step) {
        this.step = step;
    }

}
