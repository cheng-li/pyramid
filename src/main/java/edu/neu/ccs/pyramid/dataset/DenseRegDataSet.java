package edu.neu.ccs.pyramid.dataset;

import java.io.File;
import java.util.Arrays;

/**
 * Created by chengli on 8/7/14.
 */
public class DenseRegDataSet extends DenseDataSet implements RegDataSet {
    private double[] labels;

    DenseRegDataSet(int numDataPoints, int numFeatures, boolean missingValue) {
        super(numDataPoints, numFeatures, missingValue);
        this.labels = new double[numDataPoints];
    }

    @Override
    public double[] getLabels() {
        return this.labels;
    }

    @Override
    public void setLabel(int dataPointIndex, double label) {
        this.labels[dataPointIndex]=label;
    }


    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder();
        sb.append(super.toString());
        sb.append("labels = ").append(Arrays.toString(labels));
        return sb.toString();
    }

    @Override
    public String getMetaInfo() {
        StringBuilder sb = new StringBuilder();
        sb.append(super.getMetaInfo());
        sb.append("type = ").append("dense regression");
        return sb.toString();
    }
}
