package edu.neu.ccs.pyramid.core.dataset;

/**
 * Created by chengli on 8/13/14.
 */
public interface ClfDataSet extends DataSet{
    int getNumClasses();
    int[] getLabels();
    void setLabel(int dataPointIndex, int label);
    LabelTranslator getLabelTranslator();
    void setLabelTranslator(LabelTranslator labelTranslator);
}
