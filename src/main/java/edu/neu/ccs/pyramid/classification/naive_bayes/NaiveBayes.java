package edu.neu.ccs.pyramid.classification.naive_bayes;

import edu.neu.ccs.pyramid.classification.Classifier;
import edu.neu.ccs.pyramid.dataset.ClfDataSet;
import edu.neu.ccs.pyramid.dataset.DataSetUtil;
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
import java.util.List;


/**
 * Created by Rainicy on 10/8/14.
 */
public class NaiveBayes<T extends Distribution> implements Classifier.ProbabilityEstimator {

    private Class<T> clazz;

    /** Prior Probability. */
    protected double[] priors;
    protected double[] logPriors;

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
        initPriors(clfDataSet.getLabels(), clfDataSet.getNumClasses());

    }

    /**
     * Given the labels and number of classes, then initialize prior probability.
     * @param labels
     * @param numClasses
     */
    private void initPriors(int[] labels, int numClasses) {
        if (labels.length == 0) {
            throw new IllegalArgumentException("Given labels' length equals zero.");
        }
        priors = new double[numClasses];
        logPriors = new double[numClasses];
        int[] counts = new int[numClasses];
        for (int i=0; i<labels.length; i++) {
            counts[labels[i]]++;
        }
        for (int i=0; i<numClasses; i++) {
            priors[i] = (double) counts[i] / labels.length;
            logPriors[i] = Math.log(priors[i]);
        }
    }

    // change the newInstance function using Vector
    public void build(ClfDataSet clfDataSet)
            throws NoSuchMethodException, IllegalAccessException,
            InvocationTargetException, InstantiationException {
        if(this.clazz == Histogram.class) {
            throw new RuntimeException("Histogram distribution needs" +
                    "the numBins.");
        } else if (this.clazz == Multinomial.class) { // Multi initialization
            init(clfDataSet);
            // get nonzero features counts for each class(Ny)
            int[] nonzeroPerClass = new int[numClasses];
            int[] labels = clfDataSet.getLabels();
            for (int i=0; i<labels.length; i++) {
                int label = labels[i];
                int nonzeroCount = clfDataSet.getRow(i).getNumNonZeroElements();
                nonzeroPerClass[label] += nonzeroCount;
            }

            Constructor constructor = clazz.getConstructor(int.class, int.class, int.class);
            for (int i=0; i<numFeatures; i++) {
                Vector vector = clfDataSet.getColumn(i);
                List<ArrayList<Double>> classGroup = new ArrayList<>();
                for (int j=0; j<numClasses; j++) {
                    classGroup.add(new ArrayList<Double>());
                }
                for (Vector.Element element : vector.nonZeroes()) {
                    int index = element.index();
                    int label = labels[index];
                    classGroup.get(label).add(element.get());
                }
                for (int y=0; y<numClasses; y++) {
                    int Nyi = classGroup.get(y).size();
                    int Ny = nonzeroPerClass[y];
                    this.distribution[y][i] = (T)constructor.newInstance(Nyi, Ny, numFeatures);
                }
            }
        } else { // other cases
            init(clfDataSet);

            int[] countPerClass = DataSetUtil.getCountPerClass(clfDataSet);
            int[] labels = clfDataSet.getLabels();

            // initialize the specific distribution class.
            Constructor constructor = clazz.getConstructor(double[].class, int.class);
            for (int j=0; j<numFeatures; j++) {
                Vector vector = clfDataSet.getColumn(j);
                List<ArrayList<Double>> classGroup = new ArrayList<>();
                for (int i=0; i<numClasses; i++) {
                    classGroup.add(new ArrayList<Double>());
                }
                for (Vector.Element element : vector.nonZeroes()) {
                    int index = element.index();
                    int label = labels[index];
                    classGroup.get(label).add(element.get());
                }
                for (int i=0; i<numClasses; i++) {
                    double[] variables = ArrayUtils.toPrimitive(classGroup.get(i).toArray(
                            new Double[classGroup.get(i).size()]));
                    this.distribution[i][j] = (T)constructor.newInstance(variables, countPerClass[i]);
                }
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

        int[] countPerClass = DataSetUtil.getCountPerClass(clfDataSet);
        int[] labels = clfDataSet.getLabels();

        // initialize the specific distribution class.
        Constructor constructor = clazz.getConstructor(int.class, double[].class, int.class);
        for (int j=0; j<numFeatures; j++) {
            Vector vector = clfDataSet.getColumn(j);
            List<ArrayList<Double>> classGroup = new ArrayList<>();
            for (int i=0; i<numClasses; i++) {
                classGroup.add(new ArrayList<Double>());
            }
            for (Vector.Element element : vector.nonZeroes()) {
                int index = element.index();
                int label = labels[index];
                classGroup.get(label).add(element.get());
            }
            for (int i=0; i<numClasses; i++) {
                double[] variables = ArrayUtils.toPrimitive(classGroup.get(i).toArray(
                        new Double[classGroup.get(i).size()]));
                this.distribution[i][j] = (T)constructor.newInstance(numBins, variables, countPerClass[i]);
            }
        }
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
        double maxLogProb = Double.NEGATIVE_INFINITY;
        int predictLabel = -1;

        // find the maximum log probability label
        for (int i=0; i<numClasses; i++) {
            double logProb = logProbs[i];
            if (logProb > maxLogProb) {
                maxLogProb = logProb;
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
        Vector input;
        if (vector.isDense()) {
            input = vector;
        } else {
            input = new DenseVector(vector);
        }

        double[] logProbs = new double[numClasses];

        for (int label=0; label<numClasses; label++) {
            logProbs[label] = logPriors[label];
            for (int feature=0; feature<input.size(); feature++) {
                double variable = input.get(feature);
                logProbs[label] += this.distribution[label][feature].logProbability(variable);
            }
        }
        return logProbs;
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
