package edu.neu.ccs.pyramid.multilabel_classification;

import edu.neu.ccs.pyramid.dataset.MultiLabel;

/**
 * Created by chengli on 8/30/16.
 */
public interface MLScorer {
    double score(int numClasses, MultiLabel truth, MultiLabel prediction);
}
