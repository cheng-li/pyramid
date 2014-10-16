package edu.neu.ccs.pyramid.regression.prob_reg_tree;

import java.util.Arrays;

/**
 * default output of regression tree
 * Created by chengli on 8/5/14.
 */
public class AverageOutputCalculator implements LeafOutputCalculator{
    private double[] labels;

    public AverageOutputCalculator(double[] labels) {
        this.labels = labels;
    }

    @Override
    public double getLeafOutput(int[] dataAppearance) {
        return Arrays.stream(dataAppearance).mapToDouble(i -> this.labels[i])
                .average().getAsDouble();
    }
}
