package edu.neu.ccs.pyramid.regression.prob_reg_tree;

/**
 * Created by chengli on 8/5/14.
 */
class SplitResult {
    private int featureIndex;
    private double threshold;
    private double reduction;
    private int leftCount;
    private int rightCount;

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

    int getLeftCount() {
        return leftCount;
    }

    SplitResult setLeftCount(int leftCount) {
        this.leftCount = leftCount;
        return this;
    }

    int getRightCount() {
        return rightCount;
    }

    SplitResult setRightCount(int rightCount) {
        this.rightCount = rightCount;
        return this;
    }


    int getFeatureIndex() {
        return featureIndex;
    }

    double getThreshold() {
        return threshold;
    }

    double getReduction() {
        return reduction;
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
