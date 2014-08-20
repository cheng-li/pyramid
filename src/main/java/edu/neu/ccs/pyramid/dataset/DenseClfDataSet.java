package edu.neu.ccs.pyramid.dataset;

import java.io.File;
import java.util.Arrays;

/**
 * Created by chengli on 8/14/14.
 */
public class DenseClfDataSet extends DenseDataSet implements ClfDataSet{
    private int[] labels;
    public DenseClfDataSet(int numDataPoints, int numFeatures) {
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

    public static DenseClfDataSet loadStandard(File featureFile,
                                               File labelFile,
                                               String delimiter) throws Exception{
        int[] stats = DataSetUtil.parseStandard(featureFile,labelFile,delimiter);
        int numDataPoints = stats[0];
        int numFeatures = stats[1];
        System.out.println("loading data set from "+featureFile.getAbsolutePath()+
                " and "+labelFile.getAbsolutePath());
        System.out.println("number of data points = "+numDataPoints);
        System.out.println("number of features = "+numFeatures);
        DenseClfDataSet dataSet = new DenseClfDataSet(numDataPoints,numFeatures);
        DataSetUtil.loadStandard(dataSet,featureFile,labelFile,delimiter);
        System.out.println("data set loaded");
        return dataSet;
    }

    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder();
        sb.append(super.toString());
        sb.append("labels = ").append(Arrays.toString(labels));
        return sb.toString();
    }
}
