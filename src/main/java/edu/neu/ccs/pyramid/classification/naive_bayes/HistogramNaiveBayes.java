package edu.neu.ccs.pyramid.classification.naive_bayes;

import edu.neu.ccs.pyramid.classification.Classifier;
import edu.neu.ccs.pyramid.classification.ProbabilityEstimator;
import edu.neu.ccs.pyramid.dataset.FeatureRow;

import java.io.File;

/**
 * Created by Rainicy on 9/30/14.
 */
public class HistogramNaiveBayes implements Classifier, ProbabilityEstimator {
    @Override
    public int predict(FeatureRow featureRow) {
        return 0;
    }

    @Override
    public int getNumClasses() {
        return 0;
    }

    @Override
    public double[] predictClassProbs(FeatureRow featureRow) {
        return new double[0];
    }

    @Override
    public void serialize(File file) throws Exception {

    }
}
