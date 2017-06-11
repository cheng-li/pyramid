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
    private double featureSamplingRate=1;
    private int numSplitIntervals;
    private boolean usePrior;

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

    double getFeatureSamplingRate() {
        return featureSamplingRate;
    }

    int getNumSplitIntervals() {
        return numSplitIntervals;
    }


    public static class Builder {
        /**
         * required
         */
        private MultiLabelClfDataSet dataSet;

        /**
         * optional
         */
        int numLeaves = 2;
        double learningRate = 1;
        int minDataPerLeaf = 1;
        double featureSamplingRate=1;
        private int numSplitIntervals =100;
        boolean usePrior = true;

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

        public Builder minDataPerLeaf(int minDataPerLeaf) {
            this.minDataPerLeaf = minDataPerLeaf;
            return this;
        }

        public Builder featureSamplingRate(double featureSamplingRate) {
            this.featureSamplingRate = featureSamplingRate;
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
        this.featureSamplingRate = builder.featureSamplingRate;
        this.numSplitIntervals = builder.numSplitIntervals;
        this.usePrior = builder.usePrior;
    }
}
