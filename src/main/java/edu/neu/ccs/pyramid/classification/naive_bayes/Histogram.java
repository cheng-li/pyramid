package edu.neu.ccs.pyramid.classification.naive_bayes;

import edu.neu.ccs.pyramid.dataset.FeatureColumn;

/**
 * Created by Rainicy on 10/6/14.
 */
public class Histogram implements Distribution {


    @Override
    public void fit(FeatureColumn featureColumn) {

    }

    @Override
    public Double probability(double x) {
        return null;
    }

    @Override
    public Double cumulativeProbability(double x) {
        return null;
    }

    @Override
    public Double getMean() {
        return null;
    }

    @Override
    public Double getVariance() {
        return null;
    }

    @Override
    public boolean isValid() {
        return false;
    }
}
