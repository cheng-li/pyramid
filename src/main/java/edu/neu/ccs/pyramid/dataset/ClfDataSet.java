package edu.neu.ccs.pyramid.dataset;

/**
 * Created by chengli on 8/13/14.
 */
public interface ClfDataSet extends DataSet{
    int[] getLabels();
    void setLabel(int dataPointIndex, int label);
}
