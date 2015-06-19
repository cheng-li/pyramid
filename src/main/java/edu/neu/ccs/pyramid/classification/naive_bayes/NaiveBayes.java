package edu.neu.ccs.pyramid.classification.naive_bayes;

import edu.neu.ccs.pyramid.classification.Classifier;
import edu.neu.ccs.pyramid.dataset.ClfDataSet;
import edu.neu.ccs.pyramid.dataset.DataSetUtil;
import edu.neu.ccs.pyramid.dataset.LabelTranslator;
import edu.neu.ccs.pyramid.feature.FeatureList;
import edu.neu.ccs.pyramid.util.MathUtil;
import org.apache.commons.lang3.ArrayUtils;
import org.apache.mahout.math.Vector;

import java.lang.reflect.Array;
import java.lang.reflect.Constructor;
import java.lang.reflect.InvocationTargetException;
import java.util.ArrayList;
import java.util.List;
import java.util.stream.IntStream;


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


    // default values for speeding up prediction
    double[] sumZeroLogPosterior;  // for each label, sum of posterior probability by 0;
    double[][] zeroLogPosterior;   // for each label and feature, posterior probability by 0;




    public NaiveBayes(Class<T> type) {
        this.clazz = type;
    }

    // initialize the basic fields.
    private void init(ClfDataSet clfDataSet) {
        this.numClasses = clfDataSet.getNumClasses();
        this.numFeatures = clfDataSet.getNumFeatures();
        this.distribution = (T[][])Array.newInstance(clazz, numClasses, numFeatures);

        // initialize prior probabilities.
        this.priors = new double[this.numClasses];
        this.logPriors = new double[this.numClasses];
        int[] countPerClass = DataSetUtil.getCountPerClass(clfDataSet);
        for (int i=0; i<this.numClasses; i++) {
            this.priors[i] = (double) countPerClass[i] / clfDataSet.getNumDataPoints();
            this.logPriors[i] = Math.log(this.priors[i]);
        }

    }


    // change the newInstance function using Vector
    public void build(ClfDataSet clfDataSet)
            throws NoSuchMethodException, IllegalAccessException,
            InvocationTargetException, InstantiationException {
        if(this.clazz == Histogram.class) {
            throw new RuntimeException("Histogram distribution needs" +
                    "the numBins.");
        }
        init(clfDataSet);

        if (this.clazz == Multinomial.class) { // Multi initialization

            // get nonzero features counts for each class(Ny)
            int[] nonzeroPerClass = new int[numClasses];
            int[] labels = clfDataSet.getLabels();
            for (int i=0; i<labels.length; i++) {
                int label = labels[i];
                nonzeroPerClass[label] += clfDataSet.getRow(i).getNumNonZeroElements();;
            }

            Constructor constructor = clazz.getConstructor(int.class, int.class, int.class);
            IntStream.range(0, numFeatures).parallel()
                    .forEach(i -> buildOneFeatureForMulti(constructor, nonzeroPerClass,
                            getVariablesForClasses(clfDataSet.getColumn(i), labels), i));

        } else { // other cases

            int[] countPerClass = DataSetUtil.getCountPerClass(clfDataSet);
            int[] labels = clfDataSet.getLabels();

            // initialize the specific distribution class.
            Constructor constructor = clazz.getConstructor(double[].class, int.class);

            IntStream.range(0, numFeatures).parallel()
                    .forEach(j -> buildOneFeature(constructor, getVariablesForClasses(clfDataSet.getColumn(j), labels), countPerClass, j));
        }

        updateDefaultValues();
    }

    // build one feature for multinomial
    private void buildOneFeatureForMulti(Constructor constructor, int[] nonzeroPerClass, List<ArrayList<Double>> classGroup, int i) {

        for (int y=0; y<numClasses; y++) {
            int Nyi = classGroup.get(y).size();
            int Ny = nonzeroPerClass[y];
            try {
                this.distribution[y][i] = (T)constructor.newInstance(Nyi, Ny, numFeatures);
            } catch (InstantiationException e) {
                e.printStackTrace();
            } catch (IllegalAccessException e) {
                e.printStackTrace();
            } catch (InvocationTargetException e) {
                e.printStackTrace();
            }
        }
    }

    // build one feature other distributions
    private void buildOneFeature(Constructor constructor, List<ArrayList<Double>> classGroup, int[] countPerClass, int j) {

        for (int i=0; i<numClasses; i++) {
            double[] variables = ArrayUtils.toPrimitive(classGroup.get(i).toArray(
                    new Double[classGroup.get(i).size()]));
            try {
                this.distribution[i][j] = (T)constructor.newInstance(variables, countPerClass[i]);
            } catch (InstantiationException e) {
                e.printStackTrace();
            } catch (IllegalAccessException e) {
                e.printStackTrace();
            } catch (InvocationTargetException e) {
                e.printStackTrace();
            }
        }
    }


    // build features for Histogram distribution
    public void build(ClfDataSet clfDataSet, int numBins) throws NoSuchMethodException {
        if(this.clazz != Histogram.class) {
            throw new RuntimeException("Histogram distribution should be " +
                    "given.");
        }

        init(clfDataSet);

        int[] countPerClass = DataSetUtil.getCountPerClass(clfDataSet);
        int[] labels = clfDataSet.getLabels();

        // initialize the specific distribution class.
        Constructor constructor = clazz.getConstructor(int.class, double[].class, int.class);

        IntStream.range(0, numFeatures).parallel()
                .forEach(j -> buildOneFeature(constructor, getVariablesForClasses(clfDataSet.getColumn(j), labels), countPerClass, numBins, j));

        updateDefaultValues();
    }

    // build feature for Histogram distribution
    private void buildOneFeature(Constructor constructor, List<ArrayList<Double>> classGroup, int[] countPerClass, int numBins, int j) {

        for (int i=0; i<numClasses; i++) {
            double[] variables = ArrayUtils.toPrimitive(classGroup.get(i).toArray(
                    new Double[classGroup.get(i).size()]));
            try {
                this.distribution[i][j] = (T)constructor.newInstance(numBins, variables, countPerClass[i]);
            } catch (InstantiationException e) {
                e.printStackTrace();
            } catch (IllegalAccessException e) {
                e.printStackTrace();
            } catch (InvocationTargetException e) {
                e.printStackTrace();
            }
        }
    }

    // get variables for each class
    private List<ArrayList<Double>> getVariablesForClasses(Vector vector, int[] labels) {
        List<ArrayList<Double>> classGroup = new ArrayList<>();
        for (int i=0; i<this.numClasses; i++) {
            classGroup.add(new ArrayList<>());
        }
        for (Vector.Element element : vector.nonZeroes()) {
            int index = element.index();
            classGroup.get(labels[index]).add(element.get());
        }

        return classGroup;
    }


    /**
     * update the sumZeroPosterior and zeroPosterior.
     */
    private void updateDefaultValues() {
        this.sumZeroLogPosterior = new double[numClasses];
        this.zeroLogPosterior = new double[numClasses][numFeatures];

        for (int i=0; i<numClasses; i++) {
            for (int j=0; j<numFeatures; j++) {
                double zeroValue = this.distribution[i][j].logProbability(0);
                this.zeroLogPosterior[i][j] = zeroValue;
                this.sumZeroLogPosterior[i] += zeroValue;
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

        double[] logProbs = new double[numClasses];

        for (int i=0; i<numClasses; i++) {
            logProbs[i] = logPriors[i];
            logProbs[i] += sumZeroLogPosterior[i];
        }

        for (Vector.Element element : vector.nonZeroes()) {
            int index = element.index();
            double value = element.get();
            for (int i=0; i<numClasses; i++) {
                logProbs[i] += this.distribution[i][index].logProbability(value) - this.zeroLogPosterior[i][index];
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
