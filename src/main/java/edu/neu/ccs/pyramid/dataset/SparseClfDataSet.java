package edu.neu.ccs.pyramid.dataset;

import java.io.File;
import java.util.Arrays;

/**
 * Created by chengli on 8/14/14.
 */
public class SparseClfDataSet extends SparseDataSet implements ClfDataSet {
    private int[] labels;
    public SparseClfDataSet(int numDataPoints, int numFeatures) {
        super(numDataPoints, numFeatures);
        this.labels = new int[numDataPoints];
    }

    @Override
    public int[] getLabels() {
        return this.labels;
    }

    @Override
    public void setLabel(int dataPointIndex, int label) {
        this.labels[dataPointIndex]=label;
    }

    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder();
        sb.append(super.toString());
        sb.append("labels = ").append(Arrays.toString(labels));
        return sb.toString();
    }
}
