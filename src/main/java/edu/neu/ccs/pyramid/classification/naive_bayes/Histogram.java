package edu.neu.ccs.pyramid.classification.naive_bayes;

import edu.neu.ccs.pyramid.dataset.FeatureColumn;
import org.apache.mahout.math.Vector;

import java.util.Arrays;

/**
 * Created by Rainicy on 10/6/14.
 */
public class Histogram implements Distribution {


    /** Number of bins. */
    protected int bins;
    /** Values for each bin. */
    protected double[] values;

    /** Minimum value in the histogram. */
    protected double min;
    /** Maximum value in the histogram. */
    protected double max;
    /** Each step */
    protected double step;


    /** Default constructor, set bins equals 2. */
    public Histogram() {
        this(2);
    }

    /**  Constructor by given number of bins. */
    public Histogram(int bins) {
        this.bins = bins;
        values = new double[bins];
    }

    /** Constructor by given bins and variables. */
    public Histogram(int bins, double[] variables) {
        this(bins);
        fit(variables);
    }

    @Override
    public void fit(double[] variables) throws IllegalArgumentException{

        Arrays.sort(variables);
        setMin(variables[0]);
        setMax(variables[variables.length-1]);

        if (getMin() >= getMax()) {
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


        for (int i=0; i<getBins(); i++) {
            double value = ((double)counts[i])/(totalNumVariables);
            setValues(value, i);
        }
    }

    /** By given a variable, find the index bin for this variable. */
    private int getIndexOfBins(double x) {
        double distance = x - getMin();
        int index = (int) (distance / getStep());

        if (index >= getBins()-1) {
            return getBins()-1;
        }
        return index;
    }

    @Override
    public double probability(double x) {
        return getValues(getIndexOfBins(x));
    }

    @Override
    public double cumulativeProbability(double x) {
        int index = getIndexOfBins(x);
        double cum = 0;
        for (int i=0; i<=index; i++) {
            cum += getValues(i);
        }
        return cum;
    }

    /*
     TODO
      */
    @Override
    public double getMean() {
        return 0;
    }

    /*
    TODO
     */
    @Override
    public double getVariance() {
        return 0;
    }

    /*
    TODO
     */
    @Override
    public boolean isValid() {
        return false;
    }

    public String toString() {
        String str;
        str = "Total number of bins: " + getBins() + "\n";
        for (int i=0; i<getBins(); i++) {
            str += "(" + i + ") \t" + getValues(i) + "\n";
        }
        return str;
    }

    public int getBins() {
        return bins;
    }

    public void setBins(int bins) {
        this.bins = bins;
    }

    public double getValues(int index) {
        return values[index];
    }

    public void setValues(double value, int index) {
        this.values[index] = value;
    }

    public double getMax() {
        return max;
    }

    public void setMax(double max) {
        this.max = max;
    }

    public double getMin() {
        return min;
    }

    public void setMin(double min) {
        this.min = min;
    }

    public double getStep() {
        return step;
    }

    public void setStep(double step) {
        this.step = step;
    }

}
