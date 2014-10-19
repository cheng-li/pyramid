package edu.neu.ccs.pyramid.regression.regression_tree;

/**
 * Created by chengli on 8/10/14.
 */
class Interval {
    private double lower;
    private double upper;
    // probabilistic probabilisticCount
    private double probabilisticCount =0;
    // sum of labels weighted by probabilities
    private double weightedSum = 0;

    // the percentage of probability mass in this interval among total
    // probability mass at the ode
    private double percentage;

    public double getWeightedSum() {
        return weightedSum;
    }

    public void setWeightedSum(double weightedSum) {
        this.weightedSum = weightedSum;
    }

    public double getLower() {
        return lower;
    }

    public void setLower(double lower) {
        this.lower = lower;
    }

    public double getUpper() {
        return upper;
    }

    public void setUpper(double upper) {
        this.upper = upper;
    }

    public double getProbabilisticCount() {
        return probabilisticCount;
    }

    public void setProbabilisticCount(double probabilisticCount) {
        this.probabilisticCount = probabilisticCount;
    }

    public double getPercentage() {
        return percentage;
    }

    public void setPercentage(double percentage) {
        this.percentage = percentage;
    }

    @Override
    public String toString() {
        return "Interval{" +
                "lower=" + lower +
                ", upper=" + upper +
                ", probabilisticCount=" + probabilisticCount +
                ", weightedSum=" + weightedSum +
                ", percentage=" + percentage +
                '}';
    }
}
