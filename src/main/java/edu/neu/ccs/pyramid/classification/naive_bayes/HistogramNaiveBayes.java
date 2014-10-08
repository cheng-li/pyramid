package edu.neu.ccs.pyramid.classification.naive_bayes;

import edu.neu.ccs.pyramid.classification.Classifier;
import edu.neu.ccs.pyramid.classification.ProbabilityEstimator;
import edu.neu.ccs.pyramid.dataset.ClfDataSet;
import edu.neu.ccs.pyramid.dataset.DataSet;
import edu.neu.ccs.pyramid.dataset.FeatureRow;

import java.io.File;

/**
 * Created by Rainicy on 9/30/14.
 */
public class HistogramNaiveBayes implements Classifier, ProbabilityEstimator {

    /** PriorProbability class. */
    protected PriorProbability priorProbability;

    /** Histograms for each label and each feature. */
    protected Histogram[][] histograms;

    public HistogramNaiveBayes(ClfDataSet clfDataSet) {
        priorProbability = new PriorProbability(clfDataSet);

        int numClasses = clfDataSet.getNumClasses();
        int numFeatures = clfDataSet.getNumFeatures();

        histograms = new Histogram[numClasses][numFeatures];


        for (int i=0; i<numClasses; i++) {
            for (int j=0; j<numFeatures; i++) {
            }
        }
    }


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
