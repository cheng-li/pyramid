package edu.neu.ccs.pyramid.dataset;

import java.util.Collection;

/**
 * Created by chengli on 9/27/14.
 */
public interface MultiLabelClfDataSet extends DataSet{
    MultiLabel[] getMultiLabels();
    void addLabel(int dataPointIndex, int classIndex);

    default void addLabels(int dataPointIndex, Collection<Integer> labels){
        for (Integer label: labels){
            addLabel(dataPointIndex,label);
        }
    }

    int getNumClasses();
}
