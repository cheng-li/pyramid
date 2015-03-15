package edu.neu.ccs.pyramid.classification.naive_bayes;

import edu.neu.ccs.pyramid.classification.ProbabilityEstimator;
import edu.neu.ccs.pyramid.dataset.ClfDataSet;
import edu.neu.ccs.pyramid.dataset.LabelTranslator;
import edu.neu.ccs.pyramid.feature.FeatureList;
import edu.neu.ccs.pyramid.util.MathUtil;
import org.apache.commons.lang3.ArrayUtils;
import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.Vector;

import java.lang.reflect.Array;
import java.lang.reflect.Constructor;
import java.lang.reflect.InvocationTargetException;
import java.util.ArrayList;


/**
 * Created by Rainicy on 10/8/14.
 */
public class NaiveBayes<T extends Distribution> implements ProbabilityEstimator {

    private Class<T> clazz;

    /** Prior Probability. */
    protected PriorProbability priors;

    /**
     * Conditional Probability distribution for each label and
     * each feature. The total distribution size should be
     * (Number_Classes * Number_Features)
     */
    private T[][] distribution;

    protected int numClasses;
    protected int numFeatures;
    private FeatureList featureList;
    private LabelTranslator labelTranslator;


    public NaiveBayes(Class<T> type) {
        this.clazz = type;
    }

    // initialize the basic fields.
    private void init(ClfDataSet clfDataSet) {
        this.numClasses = clfDataSet.getNumClasses();
        this.numFeatures = clfDataSet.getNumFeatures();
        this.distribution = (T[][])Array.newInstance(clazz, numClasses, numFeatures);
        this.priors = new PriorProbability(clfDataSet);

    }

    public void build(ClfDataSet clfDataSet)
            throws NoSuchMethodException, IllegalAccessException,
            InvocationTargetException, InstantiationException {
        if(this.clazz == Histogram.class) {
            throw new RuntimeException("Histogram distribution needs" +
                    "the numBins.");
        }

        init(clfDataSet);

        // initialize the specific distribution class.
        Constructor constructor = clazz.getConstructor(double[].class);
        for (int i=0; i<numClasses; i++) {
            for (int j=0; j<numFeatures; j++) {
                double[] variables = getVariablesByLabelFeature(clfDataSet, i, j);
                // fitting non-histogram distribution class
                this.distribution[i][j] = (T)constructor.newInstance(variables);
            }
        }
    }

    public void build(ClfDataSet clfDataSet, int numBins)
            throws NoSuchMethodException, IllegalAccessException,
            InvocationTargetException, InstantiationException {
        if(this.clazz != Histogram.class) {
            throw new RuntimeException("Histogram distribution should be " +
                    "given.");
        }

        init(clfDataSet);

        Constructor constructor = clazz.getConstructor(int.class, double[].class);
        for (int i=0; i<numClasses; i++) {
            for (int j=0; j<numFeatures; j++) {
                double[] variables = getVariablesByLabelFeature(clfDataSet, i, j);
                // fitting the histogram distribution.
                this.distribution[i][j] = (T)constructor.newInstance(numBins, variables);
            }
        }
    }


    private double[] getVariablesByLabelFeature(ClfDataSet clfDataSet,
                                                 int label, int feature) {
        // return result
        ArrayList<Double> listVariables = new ArrayList<Double>();

        int numDataPoints = clfDataSet.getNumDataPoints();
        int[] labels = clfDataSet.getLabels();
        Vector featureVector = clfDataSet.getColumn(feature);

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
    public int predict(Vector vector) {

        double[] logProbs = predictClassLogProbs(vector);
//        System.out.println(Arrays.toString(logProbs));
//        System.out.println(Arrays.toString(logProbs));
        double maxLogProb = Double.NEGATIVE_INFINITY;
        int predictLabel = -1;

        // find the maximum log probability label
        for (int i=0; i<numClasses; i++) {
            if (logProbs[i] > maxLogProb) {
                maxLogProb = logProbs[i];
                predictLabel = i;
            }
        }

        // Cannot predict the label.
        if (predictLabel == -1) {
            throw new RuntimeException("Failed to predict the label.");
        }

        return predictLabel;
    }

    @Override
    public int getNumClasses() {
        return numClasses;
    }


    @Override
    /**
     * Calculate the posterior probability.
     * P(y|X) = P(X|y)*P(y) / P(X)
     */
    public double[] predictClassProbs(Vector vector) {
        double[] logProbs = predictClassLogProbs(vector);

        double logDenominator = MathUtil.logSumExp(logProbs);
        double[] probs = new double[logProbs.length];

        for (int i=0; i<logProbs.length; i++) {
            probs[i] = Math.exp(logProbs[i] - logDenominator);
        }

        return probs;
    }

    /**
     * Returns the posterior probabilities for each label.
     * @param vector
     * @return
     */
    private double[] predictClassLogProbs(Vector vector) {
//        DenseVector vector = new DenseVector(vector);

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

    @Override
    public FeatureList getFeatureList() {
        return featureList;
    }

    public void setFeatureList(FeatureList featureList) {
        this.featureList = featureList;
    }

    @Override
    public LabelTranslator getLabelTranslator() {
        return labelTranslator;
    }

    public void setLabelTranslator(LabelTranslator labelTranslator) {
        this.labelTranslator = labelTranslator;
    }
}
