package edu.neu.ccs.pyramid.regression.regression_tree;

import edu.neu.ccs.pyramid.dataset.DataSet;

import java.util.concurrent.ExecutorService;

/**
 * Created by chengli on 8/5/14.
 */
public class RegTreeConfig {
    private int maxNumLeaves;
    private int minDataPerLeaf;
    private DataSet dataSet;
    /**
     * fit regression tree with labels
     */
    private double[] labels;
    /**
     * features to consider in the tree
     */
    private int[] activeFeatures;
    /**
     * data points to consider in the tree
     */
    private int[] activeDataPoints;

    private LeafOutputCalculator leafOutputCalculator;

    public LeafOutputCalculator getLeafOutputCalculator() {
        return leafOutputCalculator;
    }

    public RegTreeConfig setDataSet(DataSet dataSet) {
        this.dataSet = dataSet;
        return this;
    }

    public RegTreeConfig setLabels(double[] labels) {
        this.labels = labels;
        return this;
    }

    public RegTreeConfig setLeafOutputCalculator(LeafOutputCalculator leafOutputCalculator) {
        this.leafOutputCalculator = leafOutputCalculator;
        return this;
    }

    public RegTreeConfig useDefaultOutputCalculator(){
        if (this.labels==null){
            throw new RuntimeException("labels are not assigned");
        }
        this.leafOutputCalculator = new AverageOutputCalculator(this.labels);
        return this;
    }

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

    int getMaxNumLeaves() {
        return maxNumLeaves;
    }

    int[] getActiveDataPoints() {
        return activeDataPoints;
    }

    int getMinDataPerLeaf() {
        return minDataPerLeaf;
    }

    DataSet getDataSet() {
        return dataSet;
    }

    double[] getLabels() {
        return labels;
    }

    int[] getActiveFeatures() {
        return activeFeatures;
    }

}
