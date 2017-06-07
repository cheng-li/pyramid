package edu.neu.ccs.pyramid.dataset;

import java.util.Arrays;
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

    LabelTranslator getLabelTranslator();
    void setLabelTranslator(LabelTranslator labelTranslator);

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
