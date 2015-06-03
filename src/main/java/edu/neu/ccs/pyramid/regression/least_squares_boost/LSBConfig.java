package edu.neu.ccs.pyramid.regression.least_squares_boost;

import edu.neu.ccs.pyramid.dataset.RegDataSet;
import edu.neu.ccs.pyramid.regression.regression_tree.LeafOutputType;
import edu.neu.ccs.pyramid.util.Sampling;

import java.util.stream.IntStream;

/**
 * Created by chengli on 6/3/15.
 */
public class LSBConfig {
    private RegDataSet dataSet;
    private double learningRate;
    private int numLeaves;
    private int minDataPerLeaf;
    private int[] activeFeatures;
    private int[] activeDataPoints;
    private int numSplitIntervals;
    private int randomLevel;
    private boolean considerHardTree;
    private boolean considerExpectationTree;
    private boolean considerProbabilisticTree;
    private boolean softTreeEarlyStop;

    public static Builder getBuilder(RegDataSet regDataSet){
        return new Builder(regDataSet);
    }

    RegDataSet getDataSet() {
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

    int getRandomLevel() {
        return randomLevel;
    }


    public boolean considerHardTree() {
        return considerHardTree;
    }

    public boolean considerExpectationTree() {
        return considerExpectationTree;
    }

    public boolean considerProbabilisticTree() {
        return considerProbabilisticTree;
    }

    public boolean softTreeEarlyStop() {
        return softTreeEarlyStop;
    }

    public static class Builder {
        /**
         * required
         */
        private RegDataSet dataSet;

        /**
         * optional
         */
        int numLeaves = 2;
        double learningRate = 1;
        int minDataPerLeaf = 1;
        double dataSamplingRate=1;
        double featureSamplingRate=1;
        private int numSplitIntervals =100;
        private int randomLevel =1;
        private boolean considerHardTree=true;
        private boolean considerExpectationTree=false;
        private boolean considerProbabilisticTree=false;
        private boolean softTreeEarlyStop=false;

        public Builder(RegDataSet dataSet) {
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


        public Builder randomLevel(int randomLevel){
            this.randomLevel = randomLevel;
            return this;
        }


        public Builder considerHardTree(boolean considerHardTree) {
            this.considerHardTree = considerHardTree;
            return this;
        }

        public Builder considerExpectationTree(boolean considerExpectationTree) {
            this.considerExpectationTree = considerExpectationTree;
            return this;
        }

        public Builder considerProbabilisticTree(boolean considerProbabilisticTree) {
            this.considerProbabilisticTree = considerProbabilisticTree;
            return this;
        }

        public Builder softTreeEarlyStop(boolean softTreeEarlyStop) {
            this.softTreeEarlyStop = softTreeEarlyStop;
            return this;
        }

        public LSBConfig build() {
            return new LSBConfig(this);
        }
    }



    //PRIVATE
    private LSBConfig(Builder builder) {
        this.dataSet = builder.dataSet;
        this.learningRate = builder.learningRate;
        this.numLeaves = builder.numLeaves;
        this.minDataPerLeaf = builder.minDataPerLeaf;
        double dataSamplingRate = builder.dataSamplingRate;
        double featureSamplingRate = builder.featureSamplingRate;
        this.numSplitIntervals = builder.numSplitIntervals;
        this.randomLevel = builder.randomLevel;
        int numDataPoints = dataSet.getNumDataPoints();
        this.considerHardTree=builder.considerHardTree;
        this.considerExpectationTree=builder.considerExpectationTree;
        this.considerProbabilisticTree=builder.considerProbabilisticTree;
        this.softTreeEarlyStop=builder.softTreeEarlyStop;
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
