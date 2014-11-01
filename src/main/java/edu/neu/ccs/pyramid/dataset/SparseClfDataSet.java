package edu.neu.ccs.pyramid.dataset;

import java.io.File;
import java.util.Arrays;

/**
 * Created by chengli on 8/14/14.
 */
public class SparseClfDataSet extends SparseDataSet implements ClfDataSet {
    int numClasses;
    private int[] labels;
    public SparseClfDataSet(int numDataPoints, int numFeatures,
                            boolean missingValue, int numClasses) {
        super(numDataPoints, numFeatures, missingValue);
        this.labels = new int[numDataPoints];
        this.numClasses = numClasses;
    }

    @Override
    public int getNumClasses() {
        return this.numClasses;
    }

    @Override
    public int[] getLabels() {
        return this.labels;
    }

    @Override
    public void setLabel(int dataPointIndex, int label) {
        if (label<0||label>=this.numClasses){
            throw new IllegalArgumentException("label<0||label>=this.numClasses");
        }
        this.labels[dataPointIndex]=label;
    }

    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder();
        sb.append("number of classes = ").append(this.numClasses).append("\n");
        sb.append(super.toString());
        sb.append("labels = ").append(Arrays.toString(labels));
        return sb.toString();
    }

    @Override
    public String getMetaInfo() {
        StringBuilder sb = new StringBuilder();
        sb.append(super.getMetaInfo());
        sb.append("type = ").append("sparse classification").append("\n");
        sb.append("number of classes = ").append(this.numClasses);
        return sb.toString();
    }
}
