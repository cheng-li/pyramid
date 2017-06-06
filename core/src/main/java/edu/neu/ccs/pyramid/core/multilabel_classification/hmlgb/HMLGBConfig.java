package edu.neu.ccs.pyramid.core.multilabel_classification.hmlgb;

import edu.neu.ccs.pyramid.core.util.Sampling;
import edu.neu.ccs.pyramid.core.dataset.MultiLabelClfDataSet;

import java.util.stream.IntStream;

/**
 * Created by chengli on 9/27/14.
 */
public class HMLGBConfig {
    private MultiLabelClfDataSet dataSet;
    private double learningRate;
    private int numLeaves;
    private int minDataPerLeaf;
    private int[] activeFeatures;
    private int[] activeDataPoints;
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

    int[] getActiveFeatures() {
        return activeFeatures;
    }

    void setActiveFeatures(int[] activeFeatures) {
        this.activeFeatures = activeFeatures;
    }

    int[] getActiveDataPoints() {
        return activeDataPoints;
    }

    void setActiveDataPoints(int[] activeDataPoints) {
        this.activeDataPoints = activeDataPoints;
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
        double dataSamplingRate=1;
        double featureSamplingRate=1;
        private int numSplitIntervals =100;
        boolean usePrior=true;

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


        public Builder dataSamplingRate(double dataSamplingRate) {
            this.dataSamplingRate = dataSamplingRate;
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

        public HMLGBConfig build() {
            return new HMLGBConfig(this);
        }
    }



    //PRIVATE
    private HMLGBConfig(Builder builder) {
        this.dataSet = builder.dataSet;
        this.learningRate = builder.learningRate;
        this.numLeaves = builder.numLeaves;
        this.minDataPerLeaf = builder.minDataPerLeaf;
        double dataSamplingRate = builder.dataSamplingRate;
        double featureSamplingRate = builder.featureSamplingRate;
        this.numSplitIntervals = builder.numSplitIntervals;
        this.usePrior = builder.usePrior;
        int numDataPoints = dataSet.getNumDataPoints();
        if (dataSamplingRate == 1) {
            /**
             * preserve orders (seems does not matter for data)
             */
            this.activeDataPoints = IntStream.range(0, numDataPoints).toArray();
        } else {
            /**
             * does not preserve orders
             */
            this.activeDataPoints = Sampling.sampleByPercentage(numDataPoints,
                    dataSamplingRate);
        }

        if (featureSamplingRate == 1) {
            /**
             * preserve orders
             */
            this.activeFeatures = IntStream.range(0, this.dataSet.getNumFeatures())
                    .toArray();
        } else {
            /**
             * does not preserve orders
             */
            this.activeFeatures = Sampling.sampleByPercentage(this.dataSet.getNumFeatures(),
                    featureSamplingRate);
        }
    }
}
