package edu.neu.ccs.pyramid.dataset.row;

import edu.neu.ccs.pyramid.dataset.LabelTranslator;
/**
 * Created by Rainicy on 8/31/17
 */
public class RowSparseSeqClfDataSet extends RowSparseSeqDataSet implements RowClfDataSet {
    int numClasses;
    private int[] labels;
    private LabelTranslator labelTranslator;

    public RowSparseSeqClfDataSet(int numDatapoints, int numFeatures, int numclasses) {
        super(numDatapoints, numFeatures);
        this.labels = new int[numDatapoints];
        this.numClasses = numclasses;
        this.labelTranslator = LabelTranslator.newDefaultLabelTranslator(numclasses);
    }

    @Override
    public int getNumClasses() {
        return numClasses;
    }

    @Override
    public int[] getLabels() {
        return labels;
    }

    @Override
    public void setLabel(int dataPointIndex, int label) {
        if (label<0||label>=this.numClasses){
            throw new IllegalArgumentException("label<0||label>=this.numClasses");
        }
        labels[dataPointIndex] = label;
    }

    @Override
    public LabelTranslator getLabelTranslator() {
        return labelTranslator;
    }

    @Override
    public void setLabelTranslator(LabelTranslator labelTranslator) {
        this.labelTranslator = labelTranslator;
    }

}
