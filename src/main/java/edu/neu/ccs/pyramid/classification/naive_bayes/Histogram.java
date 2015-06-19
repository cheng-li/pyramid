package edu.neu.ccs.pyramid.classification.naive_bayes;

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


    /** Default constructor, set numBins equals 2. */
    public Histogram() {
        this(2);
    }

    /**  Constructor by given numBins. */
    public Histogram(int numBins) {
        if (numBins < 1) {
            throw new RuntimeException("Number of Bins should be bigger than 0");
        }
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
    public void fit(double[] nonzeroVars, int numPerClass){

        int numOfZeroes = numPerClass - nonzeroVars.length;

        // find min and max
        double tempMin = 0;
        double tempMax =0;
        for (int i=0; i<nonzeroVars.length; i++) {
            double value = nonzeroVars[i];
            if (value < tempMin) {
                tempMin = value;
            }
            if (value > tempMax) {
                tempMax = value;
            }
        }
        this.min = tempMin;
        this.max = tempMax;

        this.step = (this.max - this.min) / this.numBins;

        int[] counts = new int[this.numBins];
        int zeroIndex = getIndexOfBins(0);
        counts[zeroIndex] += numOfZeroes;
        // value for each bin
        for (int i=0; i<nonzeroVars.length; i++) {
            int binIndex = getIndexOfBins(nonzeroVars[i]);
            counts[binIndex] += 1;
        }


        // Smoothing here.
        for (int i=0; i<this.numBins; i++) {
            this.values[i] = ((double)counts[i]+1)/(numPerClass + this.numBins);
        }
    }

    /** By given a variable, find the index bin for this variable. */
    private int getIndexOfBins(double x) {

        // out of range
        if (x <= this.min) {
            return 0;
        } else if (x >= this.max) {
            return this.numBins - 1;
        }

        double distance = x - this.min;
        int index = (int) (distance / this.step);

        return index;
    }

    @Override
    public double probability(double x) {
        return values[getIndexOfBins(x)];
    }

    @Override
    public double logProbability(double x) throws IllegalArgumentException {
//        if (x == 0) {
//            return defaultZeroLogProb;
//        }
        double value = Math.log(values[getIndexOfBins(x)]);
        return value;
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
