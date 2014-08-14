package edu.neu.ccs.pyramid.classification;

import edu.neu.ccs.pyramid.dataset.FeatureRow;

/**
 * Created by chengli on 8/14/14.
 */
public interface ProbabilityEstimator {
    int getNumClasses();
    double[] predictClassProbs(FeatureRow featureRow);

}
