package edu.neu.ccs.pyramid.regression;

import edu.neu.ccs.pyramid.dataset.FeatureRow;

/**
 * Created by chengli on 8/6/14.
 */
public interface Regressor {
    double predict(FeatureRow featureRow);
}
