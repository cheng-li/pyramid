package edu.neu.ccs.pyramid.multilabel_classification;

import edu.neu.ccs.pyramid.dataset.MultiLabel;

/**
 * Created by chengli on 8/30/16.
 */
public class AccScorer implements MLScorer{
    @Override
    public double score(int numClasses, MultiLabel truth, MultiLabel prediction) {
        if (truth.equals(prediction)){
            return 1;
        } else {
            return 0;
        }
    }
}
