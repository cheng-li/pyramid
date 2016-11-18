package edu.neu.ccs.pyramid.multilabel_classification.cbm;

import edu.neu.ccs.pyramid.classification.Classifier;
import edu.neu.ccs.pyramid.classification.lkboost.LKBoost;
import edu.neu.ccs.pyramid.dataset.LabelTranslator;
import edu.neu.ccs.pyramid.dataset.MultiLabel;
import edu.neu.ccs.pyramid.feature.FeatureList;
import edu.neu.ccs.pyramid.multilabel_classification.MultiLabelClassifier;
import edu.neu.ccs.pyramid.util.ArgSort;
import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.Vector;

import java.io.Serializable;
import java.util.Arrays;
import java.util.List;

/**
 * Created by chengli on 11/15/16.
 */
public class CBMS implements MultiLabelClassifier.ClassProbEstimator, Serializable {
    private static final long serialVersionUID = 2L;
    int numLabels;
    int numComponents;
    private int numSample = 100;
    private boolean allowEmpty = false;

    private String predictMode = "dynamic";

    // parameters
    // format: [numLabels]
    Classifier.ProbabilityEstimator[] binaryClassifiers;
    Classifier.ProbabilityEstimator multiClassClassifier;


    public CBMS(int numLabels, int numComponents) {
        this.numLabels = numLabels;
        this.numComponents = numComponents;
        this.binaryClassifiers = new LKBoost[numLabels];
        for (int l=0;l<numLabels;l++){
            binaryClassifiers[l] = new LKBoost(2);
        }


        this.multiClassClassifier = new LKBoost(numComponents);
    }

    Vector augment(Vector x, int k){
        Vector augmented = new DenseVector(x.size()+numComponents);
        for (int d=0;d<x.size();d++){
            augmented.set(d,x.get(d));
        }

        augmented.set(x.size()+k,1);
        return augmented;
    }

    @Override
    public int getNumClasses() {
        return this.numLabels;
    }


    double[] posteriorMembership(Vector x, MultiLabel y){
        BMDistribution bmDistribution = computeBM(x);
        return bmDistribution.posteriorMembership(y);
    }

    // takes time
    BMDistribution computeBM(Vector x){
        return new BMDistribution(this, x);
    }

    /**
     * for single assignment, compute log assignment probability
     * @param x
     * @param y
     * @return
     */
    private double predictLogAssignmentProb(Vector x, MultiLabel y){
        BMDistribution bmDistribution = computeBM(x);
        return bmDistribution.logProbability(y);
    }

    public double predictAssignmentProb(Vector vector, MultiLabel assignment){
        return Math.exp(predictLogAssignmentProb(vector,assignment));
    }


    /**
     * for batch jobs, use this to save computation
     * @param x
     * @param assignments
     * @return
     */
    private double[] predictLogAssignmentProbs(Vector x, List<MultiLabel> assignments){
        BMDistribution bmDistribution = computeBM(x);
        double[] probs = new double[assignments.size()];
        for (int c=0;c<assignments.size();c++){
            MultiLabel multiLabel = assignments.get(c);
            probs[c]= bmDistribution.logProbability(multiLabel);
        }
        return probs;
    }

    /**
     * for batch jobs, use this to save computation
     * @param vector
     * @param assignments
     * @return
     */
    public double[] predictAssignmentProbs(Vector vector, List<MultiLabel> assignments){
        double[] logProbs = predictLogAssignmentProbs(vector, assignments);
        return Arrays.stream(logProbs).map(Math::exp).toArray();
    }


    public MultiLabel predict(Vector vector) {

        // new a BMMPredictor
        CBMPredictor CBMPredictor = new CBMPredictor(computeBM(vector));
        CBMPredictor.setNumSamples(numSample);
        CBMPredictor.setAllowEmpty(allowEmpty);
        // samples methods
        switch (predictMode) {
            case "sampling":
                return CBMPredictor.predictBySampling();
            case "dynamic":
                return CBMPredictor.predictByDynamic();
            case "greedy":
                return CBMPredictor.predictByGreedy();
            case "hard":
                return CBMPredictor.predictByHardAssignment();
            case "marginal":
                return predictByMarginals(vector);
            default:
                throw new RuntimeException("Unknown predictMode: " + predictMode);
        }

    }

    /**
     * compute marginal probabilities P(y_l|x)
     * @param vector
     * @return
     */
    public double[] predictClassProbs(Vector vector){
        BMDistribution bmDistribution = computeBM(vector);
        return bmDistribution.marginals();
    }

    /**
     * predict Sign(E(y|x))
     * @param vector
     * @return
     */
    MultiLabel predictByMarginals(Vector vector){
        double[] probs = predictClassProbs(vector);
        MultiLabel prediction = new MultiLabel();
        for (int l=0;l<numLabels;l++){
            if (probs[l]>0.5){
                prediction.addLabel(l);
            }
        }
        return prediction;
    }

    /**
     * sort marginals, and keep top few
     * @param vector
     * @param top
     * @return
     */
    public MultiLabel predictByMarginals(Vector vector, int top){
        double[] probs = predictClassProbs(vector);
        int[] sortedIndices = ArgSort.argSortDescending(probs);
        MultiLabel prediction = new MultiLabel();
        for (int i=0;i<top;i++){
            prediction.addLabel(sortedIndices[i]);
        }
        return prediction;
    }




    public void setPredictMode(String mode) {
        this.predictMode = mode;
    }
    public void setAllowEmpty(boolean allowEmpty) {
        this.allowEmpty = allowEmpty;
    }


    @Override
    public FeatureList getFeatureList() {
        return null;
    }

    @Override
    public LabelTranslator getLabelTranslator() {
        return null;
    }

    public void setNumSample(int numSample) {
        this.numSample = numSample;
    }

    /**
     * batch sample
     * @param x
     * @param numSamples
     * @return
     */
    public List<MultiLabel> samples(Vector x, int numSamples){
        BMDistribution bmDistribution = computeBM(x);
        return bmDistribution.sample(numSamples);

    }


    public Classifier.ProbabilityEstimator[] getBinaryClassifiers() {
        return binaryClassifiers;
    }

    public Classifier.ProbabilityEstimator getMultiClassClassifier() {
        return multiClassClassifier;
    }

    public int getNumComponents() {
        return numComponents;
    }



}
