package edu.neu.ccs.pyramid.multilabel_classification.imlgb;

import edu.neu.ccs.pyramid.dataset.MultiLabelClfDataSet;
import edu.neu.ccs.pyramid.util.Sampling;

import java.util.stream.IntStream;

/**
 * Created by chengli on 10/8/14.
 */
public class IMLGBConfig {
    private MultiLabelClfDataSet dataSet;
    private double learningRate;
    private int numLeaves;
    private int minDataPerLeaf;
    private int numSplitIntervals;
    private boolean usePrior;
    private int numActiveFeatures;

    boolean usePrior() {
        return usePrior;
    }

    MultiLabelClfDataSet getDataSet() {
        return dataSet;
    }

    double getLearningRate() {
        return learningRate;
    }

    int getNumLeaves() {
        return numLeaves;
    }

    int getMinDataPerLeaf() {
        return minDataPerLeaf;
    }


    int getNumSplitIntervals() {
        return numSplitIntervals;
    }

    int getNumActiveFeatures() {
        return numActiveFeatures;
    }

    public static class Builder {
        /**
         * required
         */
        private MultiLabelClfDataSet dataSet;

        /**
         * optional
         */
        private int numLeaves = 2;
        private double learningRate = 1;
        private int minDataPerLeaf = 1;
        private int numSplitIntervals =100;
        private boolean usePrior = true;
        private int numActiveFeatures=20;

        public Builder(MultiLabelClfDataSet dataSet) {
            this.dataSet = dataSet;
        }

        public Builder numLeaves(int numLeaves){
            this.numLeaves = numLeaves;
            return this;
        }

        public Builder learningRate(double learningRate) {
            this.learningRate = learningRate;
            return this;
        }

        public Builder numActiveFeatures(int numActiveFeatures) {
            this.numActiveFeatures = numActiveFeatures;
            return this;
        }

        public Builder minDataPerLeaf(int minDataPerLeaf) {
            this.minDataPerLeaf = minDataPerLeaf;
            return this;
        }


        public Builder numSplitIntervals(int numSplitIntervals) {
            this.numSplitIntervals = numSplitIntervals;
            return this;
        }

        public Builder usePrior(boolean usePrior) {
            this.usePrior = usePrior;
            return this;
        }

        //todo add setter for active featureList

        public IMLGBConfig build() {
            return new IMLGBConfig(this);
        }
    }



    //PRIVATE
    private IMLGBConfig(Builder builder) {
        this.dataSet = builder.dataSet;
        this.learningRate = builder.learningRate;
        this.numLeaves = builder.numLeaves;
        this.minDataPerLeaf = builder.minDataPerLeaf;
        this.numSplitIntervals = builder.numSplitIntervals;
        this.usePrior = builder.usePrior;
        this.numActiveFeatures = builder.numActiveFeatures;
    }
}
