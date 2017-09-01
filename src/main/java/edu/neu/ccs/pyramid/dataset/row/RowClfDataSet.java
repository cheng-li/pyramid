package edu.neu.ccs.pyramid.dataset.row;

import edu.neu.ccs.pyramid.dataset.LabelTranslator;

/**
 * Created by Rainicy on 8/31/17
 */
public interface RowClfDataSet extends RowDataSet {
    int getNumClasses();
    int[] getLabels();
    void setLabel(int dataPointIndex, int label);
    LabelTranslator getLabelTranslator();
    void setLabelTranslator(LabelTranslator labelTranslator);
}
