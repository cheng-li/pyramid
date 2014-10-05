package edu.neu.ccs.pyramid.classification.boosting.lktb;

import edu.neu.ccs.pyramid.classification.TrainConfig;

/**
 * Created by chengli on 10/4/14.
 */
public class LKTBTrainConfig extends TrainConfig{
    private double learningRate = 0.1;
    private int numLeaves = 2;
    private int minDataPerLeaf = 1;
    private int numSplitIntervals = 100;
    private double dataSamplingRate=1;
    private double featureSamplingRate=1;
    private int numIterations = 500;
    private boolean usePrior = true;

    public LKTBTrainConfig() {
    }

    public double getLearningRate() {
        return learningRate;
    }

    public LKTBTrainConfig setLearningRate(double learningRate) {
        this.learningRate = learningRate;
        return this;
    }

    public int getNumLeaves() {
        return numLeaves;
    }

    public LKTBTrainConfig setNumLeaves(int numLeaves) {
        this.numLeaves = numLeaves;
        return this;
    }

    public int getMinDataPerLeaf() {
        return minDataPerLeaf;
    }

    public LKTBTrainConfig setMinDataPerLeaf(int minDataPerLeaf) {
        this.minDataPerLeaf = minDataPerLeaf;
        return this;
    }

    public int getNumSplitIntervals() {
        return numSplitIntervals;
    }

    public LKTBTrainConfig setNumSplitIntervals(int numSplitIntervals) {
        this.numSplitIntervals = numSplitIntervals;
        return this;
    }

    public double getDataSamplingRate() {
        return dataSamplingRate;
    }

    public LKTBTrainConfig setDataSamplingRate(double dataSamplingRate) {
        this.dataSamplingRate = dataSamplingRate;
        return this;
    }

    public double getFeatureSamplingRate() {
        return featureSamplingRate;
    }

    public LKTBTrainConfig setFeatureSamplingRate(double featureSamplingRate) {
        this.featureSamplingRate = featureSamplingRate;
        return this;
    }

    public int getNumIterations() {
        return numIterations;
    }

    public LKTBTrainConfig setNumIterations(int numIterations) {
        this.numIterations = numIterations;
        return this;
    }

    public boolean usePrior() {
        return usePrior;
    }

    public LKTBTrainConfig setUsePrior(boolean usePrior) {
        this.usePrior = usePrior;
        return this;
    }
}
