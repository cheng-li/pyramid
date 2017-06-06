package edu.neu.ccs.pyramid.core.dataset;

/**
 * Created by chengli on 8/7/14.
 */
public interface RegDataSet extends DataSet{
    public double[] getLabels();
    public void setLabel(int dataPointIndex, double label);
}
