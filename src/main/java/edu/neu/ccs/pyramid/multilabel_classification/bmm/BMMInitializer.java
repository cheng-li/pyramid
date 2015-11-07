package edu.neu.ccs.pyramid.multilabel_classification.bmm;

import edu.neu.ccs.pyramid.clustering.bmm.BMM;
import edu.neu.ccs.pyramid.dataset.MultiLabelClfDataSet;
import edu.neu.ccs.pyramid.multilabel_classification.MultiLabelSuggester;

/**
 * Created by chengli on 10/8/15.
 */
public class BMMInitializer {

    public void initialize(BMMClassifier bmmClassifier, MultiLabelClfDataSet dataSet){
        MultiLabelSuggester suggester = new MultiLabelSuggester(dataSet,bmmClassifier.numClusters);
        BMM bmm = suggester.getBmm();
        bmmClassifier.distributions = bmm.getDistributions();
    }
}
