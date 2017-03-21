package edu.neu.ccs.pyramid.regression.regression_tree;

/**
 * Created by chengli on 8/5/14.
 */
public class RegTreeConfig {
    private int numSplitIntervals=100;
    private int maxNumLeaves=2;
    private int minDataPerLeaf=0;
    private int randomLevel=1;

    public RegTreeConfig setMaxNumLeaves(int maxNumLeaves) {
        this.maxNumLeaves = maxNumLeaves;
        return this;
    }

    public RegTreeConfig setMinDataPerLeaf(int minDataPerLeaf) {
        this.minDataPerLeaf = minDataPerLeaf;
        return this;
    }


    public RegTreeConfig setNumSplitIntervals(int numSplitIntervals) {
        if (numSplitIntervals<=1){
            throw new IllegalArgumentException("numSplitIntervals must be greater than 2");
        }
        this.numSplitIntervals = numSplitIntervals;
        return this;
    }

    public RegTreeConfig setRandomLevel(int randomLevel) {
        if (randomLevel<1){
            throw new IllegalArgumentException("random level should be at least 1");
        }
        this.randomLevel = randomLevel;
        return this;
    }

    int getMaxNumLeaves() {
        return maxNumLeaves;
    }



    int getMinDataPerLeaf() {
        return minDataPerLeaf;
    }


    int getNumSplitIntervals() {
        return numSplitIntervals;
    }

    int getRandomLevel() {
        return randomLevel;
    }

}
