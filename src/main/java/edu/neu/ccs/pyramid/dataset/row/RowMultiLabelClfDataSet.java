package edu.neu.ccs.pyramid.dataset.row;

import edu.neu.ccs.pyramid.dataset.MultiLabel;

import java.util.Arrays;
import java.util.Collection;

/**
 * Created by Rainicy on 8/31/17
 */
public interface RowMultiLabelClfDataSet extends RowDataSet{
    MultiLabel[] getMultiLabels();
    void addLabel(int dataPointIndex, int classIndex);

    default void addLabels(int dataPointIndex, Collection<Integer> labels){
        for (Integer label: labels){
            addLabel(dataPointIndex,label);
        }
    }

    void setLabels(int dataPointIndex, MultiLabel multiLabel);

    int getNumClasses();

    /**
     * Label cardinality of a dataset D is the average number of labels of the examples in D.
     * From "Mining Multi-label Data by Grigorios Tsoumakas".
     * @return
     */
    default double labelCardinality() {
        MultiLabel[] multiLabels = getMultiLabels();
        return Arrays.stream(multiLabels).parallel().mapToDouble(multiLabel -> multiLabel.getMatchedLabels().size()).average().getAsDouble();
    };

    /**
     * Label density of D is the average number of labels of the examples in D divided by q.
     * q is the number of unique set of labels.
     * From "Mining Multi-label Data by Grigorios Tsoumakas".
     * @return
     */
    default double labelDensity() {
        return labelCardinality() / getNumClasses();
    }

}


