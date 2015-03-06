package edu.neu.ccs.pyramid.dataset;

/**
 * Created by chengli on 8/19/14.
 */
@Deprecated
public interface RankDataSet extends DataSet{
    int[] getLabels();
    void setLabel(int dataPointIndex, int label);
}
