package edu.neu.ccs.pyramid.regression.regression_tree;

/**
 * Created by chengli on 8/5/14.
 */
public class SplitResult {
    private int featureIndex;
    private double threshold;
    private double reduction;
    private double leftCount;
    private double rightCount;
    private boolean monotonic;

    SplitResult() {
    }



    SplitResult setFeatureIndex(int featureIndex) {
        this.featureIndex = featureIndex;
        return this;
    }

    SplitResult setThreshold(double threshold) {
        this.threshold = threshold;
        return this;
    }

    SplitResult setReduction(double reduction) {
        this.reduction = reduction;
        return this;
    }

    double getLeftCount() {
        return leftCount;
    }

    SplitResult setLeftCount(double leftCount) {
        this.leftCount = leftCount;
        return this;
    }

    double getRightCount() {
        return rightCount;
    }

    SplitResult setRightCount(double rightCount) {
        this.rightCount = rightCount;
        return this;
    }


    public int getFeatureIndex() {
        return featureIndex;
    }

    double getThreshold() {
        return threshold;
    }

    public double getReduction() {
        return reduction;
    }

    public boolean isMonotonic() {
        return monotonic;
    }

    public void setMonotonic(boolean monotonic) {
        this.monotonic = monotonic;
    }

    @Override
    public String toString() {
        return "SplitResult{" +
                "featureIndex=" + featureIndex +
                ", threshold=" + threshold +
                ", reduction=" + reduction +
                ", leftCount=" + leftCount +
                ", rightCount=" + rightCount +
                '}';
    }
}
