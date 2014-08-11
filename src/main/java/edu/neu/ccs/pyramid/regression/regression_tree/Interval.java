package edu.neu.ccs.pyramid.regression.regression_tree;

/**
 * Created by chengli on 8/10/14.
 */
class Interval {
    private double lower;
    private double upper;
    private int count=0;
    private double sum=0;

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

    public int getCount() {
        return count;
    }

    public void setCount(int count) {
        this.count = count;
    }

    public double getSum() {
        return sum;
    }

    public void setSum(double sum) {
        this.sum = sum;
    }

    void incrementSum(double amount){
        this.sum += amount;
    }

    void incrementCount(){
        this.count += 1;
    }

    @Override
    public String toString() {
        return "Interval{" +
                "lower=" + lower +
                ", upper=" + upper +
                ", count=" + count +
                ", sum=" + sum +
                '}';
    }
}
