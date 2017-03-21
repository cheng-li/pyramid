package edu.neu.ccs.pyramid.multilabel_classification.cbm;

import edu.neu.ccs.pyramid.dataset.MultiLabelClfDataSet;

/**
 * Created by chengli on 3/21/17.
 */
public class LRCBMOptimizer extends AbstractCBMOptimizer {

    public LRCBMOptimizer(CBM cbm, MultiLabelClfDataSet dataSet) {
        super(cbm, dataSet);
    }

    @Override
    protected void updateBinaryClassifier(int component, int label) {

    }

    @Override
    protected void updateMultiClassClassifier() {

    }

    @Override
    protected double binaryObj(int clusterIndex, int classIndex) {
        return 0;
    }

    @Override
    protected double multiClassClassifierObj() {
        return 0;
    }
}
