package edu.neu.ccs.pyramid.dataset;

/**
 * Created by chengli on 9/27/14.
 */
public interface MultiLabelClfDataSet extends DataSet{
    MultiLabel[] getMultiLabels();
    void addLabel(int dataPointIndex, int classIndex);
    int getNumClasses();
}
