package edu.neu.ccs.pyramid.regression.regression_tree;

/**
 * Created by chengli on 8/5/14.
 */
class SplitResult {
    private int featureIndex;
    private double threshold;
    private double reduction;
    private boolean valid = true;

    SplitResult(int featureIndex, double threshold, double reduction) {
        this.featureIndex = featureIndex;
        this.threshold = threshold;
        this.reduction = reduction;
    }


    boolean isValid() {
        return valid;
    }

    void setValid(boolean valid) {
        this.valid = valid;
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
                ", valid=" + valid +
                '}';
    }
}
