package edu.neu.ccs.pyramid.regression.prob_reg_tree;

/**
 * Created by chengli on 8/5/14.
 */
public class RegTreeConfig {
    private int numSplitIntervals=100;
    private int maxNumLeaves=2;
    private int minDataPerLeaf=1;

    /**
     * features to consider in the tree
     */
    private int[] activeFeatures;
    /**
     * data points to consider in the tree
     */
    private int[] activeDataPoints;

    public RegTreeConfig setActiveFeatures(int[] activeFeatures) {
        this.activeFeatures = activeFeatures;
        return this;
    }

    public RegTreeConfig setMaxNumLeaves(int maxNumLeaves) {
        this.maxNumLeaves = maxNumLeaves;
        return this;
    }

    public RegTreeConfig setMinDataPerLeaf(int minDataPerLeaf) {
        this.minDataPerLeaf = minDataPerLeaf;
        return this;
    }

    public RegTreeConfig setActiveDataPoints(int[] activeDataPoints) {
        this.activeDataPoints = activeDataPoints;
        return this;
    }

    public RegTreeConfig setNumSplitIntervals(int numSplitIntervals) {
        if (numSplitIntervals<=1){
            throw new IllegalArgumentException("numSplitIntervals must be greater than 2");
        }
        this.numSplitIntervals = numSplitIntervals;
        return this;
    }

    int getMaxNumLeaves() {
        return maxNumLeaves;
    }

    int[] getActiveDataPoints() {
        return activeDataPoints;
    }

    int getMinDataPerLeaf() {
        return minDataPerLeaf;
    }

    int[] getActiveFeatures() {
        return activeFeatures;
    }

    int getNumSplitIntervals() {
        return numSplitIntervals;
    }

}
