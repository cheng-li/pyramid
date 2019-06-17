package edu.neu.ccs.pyramid.regression.regression_tree;

import java.util.List;
import java.util.Optional;

/**
 * Created by chengli on 8/5/14.
 */
public class RegTreeConfig {
    private int numSplitIntervals=100;
    private int maxNumLeaves=2;
    private int minDataPerLeaf=0;
    private boolean parallel=true;
    private int numActiveFeatures=10;
    private String monotonicityType="none";


    public RegTreeConfig setMaxNumLeaves(int maxNumLeaves) {
        this.maxNumLeaves = maxNumLeaves;
        return this;
    }

    public RegTreeConfig setMinDataPerLeaf(int minDataPerLeaf) {
        this.minDataPerLeaf = minDataPerLeaf;
        return this;
    }

    public RegTreeConfig setParallel(boolean parallel) {
        this.parallel = parallel;
        return this;
    }

    public RegTreeConfig setNumSplitIntervals(int numSplitIntervals) {
        if (numSplitIntervals<=1){
            throw new IllegalArgumentException("numSplitIntervals must be greater than 2");
        }
        this.numSplitIntervals = numSplitIntervals;
        return this;
    }

    public RegTreeConfig setNumActiveFeatures(int numActiveFeatures){
        this.numActiveFeatures = numActiveFeatures;
        return this;
    }

    public RegTreeConfig setMonotonicityType(String monotonicityType) {
        this.monotonicityType = monotonicityType;
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

    public int getNumActiveFeatures() {
        return numActiveFeatures;
    }

    public boolean isParallel() {
        return parallel;
    }


    public String getMonotonicityType() {
        return monotonicityType;
    }
}
