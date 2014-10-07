package edu.neu.ccs.pyramid.classification.naive_bayes;

import edu.neu.ccs.pyramid.dataset.FeatureColumn;
import org.apache.mahout.math.Vector;

/**
 * Created by Rainicy on 10/6/14.
 */
public class Histogram implements Distribution {

    protected class Interval {
        double lower;
        double upper;
        double value;

        public Interval(double lower, double upper) {
            this.lower = lower;
            this.upper = upper;
        }

        public double getLower() {
            return lower;
        }

        public double getUpper() {
            return upper;
        }

        public double getValue() {
            return value;
        }

        public void setLower(double lower) {
            this.lower = lower;
        }

        public void setUpper(double upper) {
            this.upper = upper;
        }

        public void setValue(double value) {
            this.value = value;
        }
    }

    /** Number of bins. */
    protected int bins;
    /** Histogram units. */
    protected Interval[] units;


    /** Default constructor, set bins equals 2. */
    public Histogram() {
        this(2);
    }

    /**  Constructor by given number of bins. */
    public Histogram(int bins) {
        this.bins = bins;
    }

    /** Constructor by given bins and variables. */
    public Histogram(int bins, FeatureColumn featureColumn) {
        this(bins);
        fit(featureColumn);
    }

    @Override
    public void fit(FeatureColumn featureColumn) throws IllegalArgumentException{

        this.units = new Interval[bins];    // initialize the bins;

        double max;
        double min;
        max = featureColumn.getVector().maxValue();
        min = featureColumn.getVector().minValue();

        if (min >= max) {
            throw new IllegalArgumentException("Minimum value" +
                    " should be smaller than Maximum");
        }

        double step = (double)(max-min)/(getBins());
        double start = min;

        // lower and upper for each bin
        for (int i=0; i<getBins(); i++) {
            Interval unit;
            if (i == 0) {   // the first bin
                unit = new Interval(Double.MIN_VALUE, start+step);
            }
            else if (i==getBins()-1) {  // the last bin
                unit = new Interval(start, Double.MAX_VALUE);
            }
            else {
                unit = new Interval(start, start+step);
            }
            start += step;
            units[i] = unit;
        }

        // value for each bin
        int[] counts = new int[getBins()];
        Vector featureVector = featureColumn.getVector();
        int totalNumVariables = featureVector.size();
        for (int i=0; i<totalNumVariables; i++) {
            int binIndex = getIndexOfBins(featureVector.get(i));
            counts[binIndex] += 1;
        }

        for (int i=0; i<getBins(); i++) {
            double value = (double)(counts[i]/totalNumVariables);
            units[i].setValue(value);
        }
    }

    /** By given a variable, find the index bin for this variable. */
    private int getIndexOfBins(double x) {
        int i = 0;
        for ( ; i<getBins(); i++) {
            double lower = units[i].getLower();
            double upper = units[i].getUpper();

            if ((x>lower) && (x<=upper)) {
                break;
            }
        }
        return i;
    }

    @Override
    public double probability(double x) {
        return units[getIndexOfBins(x)].getValue();
    }

    @Override
    public double cumulativeProbability(double x) {
        int index = getIndexOfBins(x);
        double cum = 0;
        for (int i=0; i<=index; i++) {
            cum += units[i].getValue();
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

    public int getBins() {
        return bins;
    }

}
