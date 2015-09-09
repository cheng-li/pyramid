package edu.neu.ccs.pyramid.multilabel_classification.sampling;

import edu.neu.ccs.pyramid.dataset.MultiLabelClfDataSet;
import edu.neu.ccs.pyramid.regression.regression_tree.LeafOutputType;
import edu.neu.ccs.pyramid.util.Sampling;

import java.util.stream.IntStream;

/**
 * Created by Rainicy on 9/7/15.
 */
public class GibbsSamplingConfig {

    private MultiLabelClfDataSet dataSet;

    private double learningRate;
    private int numLeaves;
    private int minDataPerLeaf;
    private int[] activeFeatures;
    private int[] activeDataPoints;
    private int numSplitIntervals;
    private int randomLevel;
    private LeafOutputType leafOutputType;
    private boolean considerHardTree;
    private boolean considerExpectationTree;
    private boolean considerProbabilisticTree;
    private int numRounds;

    // private
    public GibbsSamplingConfig(Builder builder) {
        this.dataSet = builder.dataSet;
        this.learningRate = builder.learningRate;
        this.numLeaves = builder.numLeaves;
        this.minDataPerLeaf = builder.minDataPerLeaf;
        double dataSamplingRate = builder.dataSamplingRate;
        double featureSamplingRate = builder.featureSamplingRate;
        this.numSplitIntervals = builder.numSplitIntervals;
        this.randomLevel = builder.randomLevel;
        int numDataPoints = dataSet.getNumDataPoints();
        this.leafOutputType = builder.leafOutputType;
        this.considerHardTree=builder.considerHardTree;
        this.considerExpectationTree=builder.considerExpectationTree;
        this.considerProbabilisticTree=builder.considerProbabilisticTree;
        this.numRounds=builder.numRounds;
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

    MultiLabelClfDataSet getDataSet() {
        return dataSet;
    }

    public double getLearningRate() {
        return learningRate;
    }

    public int getNumLeaves() {
        return numLeaves;
    }

    public int getMinDataPerLeaf() {
        return minDataPerLeaf;
    }

    public int[] getActiveFeatures() {
        return activeFeatures;
    }

    public int[] getActiveDataPoints() {
        return activeDataPoints;
    }

    public int getNumSplitIntervals() {
        return numSplitIntervals;
    }

    public int getRandomLevel() {
        return randomLevel;
    }

    public LeafOutputType getLeafOutputType() {
        return leafOutputType;
    }

    public boolean isConsiderHardTree() {
        return considerHardTree;
    }

    public boolean isConsiderExpectationTree() {
        return considerExpectationTree;
    }

    public boolean isConsiderProbabilisticTree() {
        return considerProbabilisticTree;
    }

    public int getNumRounds() {
        return numRounds;
    }


    public static class Builder {
        /**
         * required
         */
        private MultiLabelClfDataSet dataSet;

        /**
         * optional
         */
        int numRounds = 100;
        int numLeaves = 2;
        double learningRate = 1;
        int minDataPerLeaf = 1;
        double dataSamplingRate=1;
        double featureSamplingRate=1;
        private int numSplitIntervals =100;
        private int randomLevel =1;
        private LeafOutputType leafOutputType = LeafOutputType.NEWTON;
        private boolean considerHardTree=true;
        private boolean considerExpectationTree=false;
        private boolean considerProbabilisticTree=false;

        public Builder(MultiLabelClfDataSet dataSet) {
            this.dataSet = dataSet;
        }

        public Builder numRounds(int numRounds) {
            this.numRounds = numRounds;
            return this;
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

        public Builder setLeafOutputType(LeafOutputType leafOutputType) {
            this.leafOutputType = leafOutputType;
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

        public GibbsSamplingConfig build() {
            return new GibbsSamplingConfig(this);
        }
    }
}
