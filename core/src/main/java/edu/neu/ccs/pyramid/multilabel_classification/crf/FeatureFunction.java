package edu.neu.ccs.pyramid.multilabel_classification.crf;

import edu.neu.ccs.pyramid.dataset.MultiLabel;
import org.apache.mahout.math.Vector;

/**
 * Created by chengli on 12/27/15.
 */
public interface FeatureFunction {
    double value(Vector vector, MultiLabel multiLabel);
}
