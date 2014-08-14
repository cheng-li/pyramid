package edu.neu.ccs.pyramid.classification.boosting.lktb;

import edu.neu.ccs.pyramid.dataset.ClfDataSet;
import edu.neu.ccs.pyramid.util.Sampling;

import java.util.concurrent.ExecutorService;
import java.util.stream.IntStream;

/**
 * Created by chengli on 8/14/14.
 */
public class LKTBConfig {
    private ClfDataSet dataSet;
    private double learningRate;
    private int numLeaves;
    private int minDataPerLeaf;
    private double dataSamplingRate;
    private double featureSamplingRate;
    private int numClasses;
    private int[] activeFeatures;
    private int[] activeDataPoints;


    ClfDataSet getDataSet() {
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

    int getNumClasses() {
        return numClasses;
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

    public static class Builder {
        /**
         * required
         */
        private ClfDataSet dataSet;
        private int numClasses;

        /**
         * optional
         */
        int numLeaves = 2;
        double learningRate = 1;
        int minDataPerLeaf = 1;
        double dataSamplingRate=1;
        double featureSamplingRate=1;

        public Builder(ClfDataSet dataSet, int numClasses) {
            this.dataSet = dataSet;
            this.numClasses = numClasses;
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

        public LKTBConfig build() {
            return new LKTBConfig(this);
        }
    }



    //PRIVATE
    private LKTBConfig(Builder builder) {
        this.dataSet = builder.dataSet;
        this.learningRate = builder.learningRate;
        this.numLeaves = builder.numLeaves;
        this.minDataPerLeaf = builder.minDataPerLeaf;
        this.dataSamplingRate = builder.dataSamplingRate;
        this.featureSamplingRate = builder.featureSamplingRate;
        this.numClasses = builder.numClasses;
        int numDataPoints = dataSet.getNumDataPoints();
        if (this.dataSamplingRate == 1) {
            /**
             * preserve orders (seems does not matter for data)
             */
            this.activeDataPoints = IntStream.range(0, numDataPoints).toArray();
        } else {
            /**
             * does not preserve orders
             */
            this.activeDataPoints = Sampling.sampleByPercentage(numDataPoints,
                    this.dataSamplingRate);
        }

        if (this.featureSamplingRate == 1) {
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
                    this.featureSamplingRate);
        }
    }
}
