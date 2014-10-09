package edu.neu.ccs.pyramid.classification.naive_bayes;

import edu.neu.ccs.pyramid.classification.Classifier;
import edu.neu.ccs.pyramid.classification.ProbabilityEstimator;
import edu.neu.ccs.pyramid.dataset.ClfDataSet;
import edu.neu.ccs.pyramid.dataset.FeatureRow;
import edu.neu.ccs.pyramid.util.MathUtil;
import org.apache.commons.lang3.ArrayUtils;
import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.Vector;

import java.util.ArrayList;

import static edu.neu.ccs.pyramid.classification.naive_bayes.DistributionType.*;

/**
 * Created by Rainicy on 10/8/14.
 */
public class NaiveBayes implements ProbabilityEstimator {

    /** Prior Probability. */
    protected PriorProbability priors;

    /**
     * Conditional Probability distribution for each label and
     * each feature. The total distribution size should be
     * (Number_Classes * Number_Features)
     */
    protected Distribution[][] distribution;
    protected DistributionType distributionType;

    protected int numClasses;
    protected int numFeatures;


    /** Constructor of not Histogram Naive Bayes. */
    public NaiveBayes(ClfDataSet clfDataSet, DistributionType type) {
        if (type == HISTOGRAM) {
            throw new IllegalArgumentException("Given distribution should " +
                    "not be a histogram.");
        }

        this.numClasses = clfDataSet.getNumClasses();
        this.numFeatures = clfDataSet.getNumFeatures();
        this.distributionType = type;
        this.priors = new PriorProbability(clfDataSet);

        switch (type) {
            case GAUSSIAN:
                System.out.println("Gaussian Distribution.");
                this.distribution = new Gaussian[numClasses][numFeatures];
                setupDistribution(clfDataSet, GAUSSIAN);
                break;
            case GAMMA:
                System.out.println("Gamma Distribution.");
                this.distribution = new Gamma[numClasses][numFeatures];
                setupDistribution(clfDataSet, GAMMA);
                break;
        }

    }

    /** Constructor of Histogram Naive Bayes. */
    public NaiveBayes(ClfDataSet clfDataSet, DistributionType type, int numBins) {
        if (type != HISTOGRAM) {
            throw new IllegalArgumentException("Given distribution should " +
                    "be a histogram.");
        }

        this.numClasses = clfDataSet.getNumClasses();
        this.numFeatures = clfDataSet.getNumFeatures();
        this.distributionType = type;
        this.priors = new PriorProbability(clfDataSet);
        this.distribution = new Histogram[numClasses][numFeatures];

        // setupDistribution.
        setupDistribution(clfDataSet, numBins);
    }

    /** Setup the distributions by given classification dataset and numBins. */
    private void setupDistribution(ClfDataSet clfDataSet, int numBins) {

        for (int i=0; i<numClasses; i++) {
            for (int j=0; j<numFeatures; j++) {
                double[] variables = getVariablesByLabelFeature(
                        clfDataSet, i, j);

                // fitting the histogram distribution
                this.distribution[i][j] = new Histogram(numBins, variables);
            }
        }
    }

    /**
     * Setup the distributions by given classification dataset and distribution type
     * */
    private void setupDistribution(ClfDataSet clfDataSet, DistributionType type) {
        switch (type) {
            case GAUSSIAN:
            {
                for (int i=0; i<numClasses; i++) {
                    for (int j=0; j<numFeatures; j++) {
                        double[] variables = getVariablesByLabelFeature(clfDataSet, i, j);

                        // fitting the Gaussian
                        this.distribution[i][j] = new Gaussian(variables);
                    }
                }
                break;
            }
            case GAMMA:
            {
                for (int i=0; i<numClasses; i++) {
                    for (int j=0; j<numFeatures; j++) {
                        double[] variables = getVariablesByLabelFeature(clfDataSet, i, j);

                        // fitting the Gamma
                        this.distribution[i][j] = new Gamma(variables);
                    }
                }
                break;
            }
        }

    }



    private double[] getVariablesByLabelFeature(ClfDataSet clfDataSet,
                                                 int label, int feature) {
        // return result
        ArrayList<Double> listVariables = new ArrayList<Double>();

        int numDataPoints = clfDataSet.getNumDataPoints();
        int[] labels = clfDataSet.getLabels();
        Vector featureVector = clfDataSet.getFeatureColumn(feature).getVector();

        for (int i=0; i<numDataPoints; i++) {
            // label matches.
            if (label == labels[i]) {
                listVariables.add(featureVector.get(i));
            }
        }

        // change to the array double and return.
        return ArrayUtils.toPrimitive(listVariables.toArray(
                new Double[listVariables.size()]
        ));
    }

    @Override
    /**
     * Calculate the log probability as the posterior
     * probability.
     *
     * e.g.
     * log(p(y|X)) = log(p(y)) + log(p(X|y))
     *
     */
    public int predict(FeatureRow featureRow) {

        double[] logProbs = predictClassLogProbs(featureRow);

        double maxLogProb = Double.NEGATIVE_INFINITY;
        int predictLabel = -1;

        // find the maximum log probability label
        for (int i=0; i<numClasses; i++) {
            if (logProbs[i] > maxLogProb) {
                maxLogProb = logProbs[i];
                predictLabel = i;
            }
        }

        return predictLabel;
    }

    @Override
    public int getNumClasses() {
        return this.numClasses;
    }

    @Override
    /**
     * Calculate the posterior probability.
     * P(y|X) = P(X|y)*P(y) / P(X)
     *        = exp(Log)
     */
    public double[] predictClassProbs(FeatureRow featureRow) {
        double[] logProbs = predictClassLogProbs(featureRow);

        double logDenominator = MathUtil.logSumExp(logProbs);
        double[] probs = new double[logProbs.length];

        for (int i=0; i<logProbs.length; i++) {
            probs[i] = Math.exp(logProbs[i] - logDenominator);
        }

        return probs;
    }

    /**
     * Returns the posterior probabilities for each label.
     * @param featureRow
     * @return
     */
    private double[] predictClassLogProbs(FeatureRow featureRow) {
        DenseVector vector = new DenseVector(featureRow.getVector());

        double[] logProbs = new double[numClasses];

        for (int label=0; label<numClasses; label++) {
            logProbs[label] = priors.logPriorProb(label);
            for (int feature=0; feature<vector.size(); feature++) {
                double variable = vector.get(feature);
                logProbs[label] += getDistribution(label,
                        feature).logProbability(variable);
            }
        }
        return logProbs;
    }

    /**
     * Returns the distribution by given label and feature.
     * @param label
     * @param feature
     * @return
     */
    private Distribution getDistribution(int label, int feature) {
        return this.distribution[label][feature];
    }
}
