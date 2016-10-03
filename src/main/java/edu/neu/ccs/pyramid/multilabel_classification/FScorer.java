package edu.neu.ccs.pyramid.multilabel_classification;

import edu.neu.ccs.pyramid.dataset.MultiLabel;
import edu.neu.ccs.pyramid.eval.InstanceAverage;

/**
 * Created by chengli on 8/30/16.
 */
public class FScorer implements MLScorer {
    @Override
    public double score(int numClaases, MultiLabel truth, MultiLabel prediction) {
        return new InstanceAverage(numClaases, truth, prediction).getF1();
    }
}
