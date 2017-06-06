package edu.neu.ccs.pyramid.core.multilabel_classification.cbm;

import edu.neu.ccs.pyramid.core.classification.lkboost.LKBoost;
import edu.neu.ccs.pyramid.core.classification.logistic_regression.LogisticRegression;
import edu.neu.ccs.pyramid.core.dataset.MultiLabel;
import edu.neu.ccs.pyramid.core.feature.FeatureList;
import edu.neu.ccs.pyramid.core.multilabel_classification.MultiLabelClassifier;
import edu.neu.ccs.pyramid.core.dataset.LabelTranslator;
import edu.neu.ccs.pyramid.core.classification.Classifier.ProbabilityEstimator;
import org.apache.mahout.math.RandomAccessSparseVector;
import org.apache.mahout.math.Vector;

import java.io.*;
import java.util.*;
import java.util.stream.Collectors;


/**
 * Conditional Bernoulli Mixtures for Multi-label Classification.
 * Cheng Li, Bingyu Wang, Virgil Pavlu, and Javed Aslam.
 * In Proceedings of the 33rd International Conference on Machine Learning (ICML), 2016.
 * Created by Rainicy on 10/23/15.
 */
public class CBM implements MultiLabelClassifier.ClassProbEstimator, Serializable {
    private static final long serialVersionUID = 2L;
    int numLabels;
    int numComponents;
    private int numFeatures;
    private int numSample = 100;
    private boolean allowEmpty = false;

    private String predictMode = "dynamic";

    private List<MultiLabel> support;

    // parameters
    // format: [numComponents][numLabels]
    ProbabilityEstimator[][] binaryClassifiers;
    ProbabilityEstimator multiClassClassifier;
    private String binaryClassifierType;
    private String multiClassClassifierType;

    private CBM() {
    }

    public String getBinaryClassifierType() {
        return binaryClassifierType;
    }

    public String getMultiClassClassifierType() {
        return multiClassClassifierType;
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
    private BMDistribution computeBM(Vector x){
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
    public List<Double> predictLogAssignmentProbs(Vector x, List<MultiLabel> assignments){
        BMDistribution bmDistribution = computeBM(x);
        List<Double> probs = new ArrayList<>();
        for (MultiLabel multiLabel: assignments){
            probs.add(bmDistribution.logProbability(multiLabel));
        }
        return probs;
    }

    /**
     * for batch jobs, use this to save computation
     * @param vector
     * @param assignments
     * @return
     */
    public List<Double> predictAssignmentProbs(Vector vector, List<MultiLabel> assignments){
        List<Double> logProbs = predictLogAssignmentProbs(vector, assignments);
        return logProbs.stream().map(Math::exp).collect(Collectors.toList());
    }


    public MultiLabel predict(Vector vector) {
        switch (predictMode) {
            case "support":
                return predictBySupport(vector);
            case "marginal":
                return predictByMarginals(vector);
        }


        // new a BMMPredictor
        CBMPredictor CBMPredictor = new CBMPredictor(vector, multiClassClassifier, binaryClassifiers, numComponents, numLabels);
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
            default:
                throw new RuntimeException("Unknown predictMode: " + predictMode);
        }

    }

    private MultiLabel predictBySupport(Vector vector) {
        List<Double> supportLogProbs = predictLogAssignmentProbs(vector, support);
        MultiLabel pred = new MultiLabel();
        double maxLogProb = Double.NEGATIVE_INFINITY;
        for (int i=0; i<supportLogProbs.size(); i++) {
            double logProb = supportLogProbs.get(i);
            if (logProb > maxLogProb) {
                maxLogProb = logProb;
                pred = support.get(i);
            }
        }
        return pred;
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


    public String toString() {
        Vector vector = new RandomAccessSparseVector(numFeatures);
        double[] mixtureCoefficients = multiClassClassifier.predictClassProbs(vector);
        final StringBuilder sb = new StringBuilder("CBM{\n");
        sb.append("numLabels=").append(numLabels).append("\n");
        sb.append("numComponents=").append(numComponents).append("\n");
        for (int k = 0; k< numComponents; k++){
            sb.append("cluster ").append(k).append(":\n");
            sb.append("proportion = ").append(mixtureCoefficients[k]).append("\n");
        }

        sb.append("multi-class component = \n");
        sb.append(multiClassClassifier);
        sb.append("binary components = \n");
        for (int k = 0; k< numComponents; k++){
            for (int l=0;l<numLabels;l++){
                sb.append("component ").append(k).append(" class ").append(l).append("\n");
                sb.append(binaryClassifiers[k][l]).append("\n");
            }
        }
        sb.append('}');
        return sb.toString();
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


    public ProbabilityEstimator[][] getBinaryClassifiers() {
        return binaryClassifiers;
    }

    public ProbabilityEstimator getMultiClassClassifier() {
        return multiClassClassifier;
    }

    public int getNumComponents() {
        return numComponents;
    }

    public static Builder getBuilder(){
        return new Builder();
    }

    public static class Builder {
        private int numClasses;
        private int numComponents;
        private int numFeatures;
        private List<MultiLabel> support;
        private String binaryClassifierType= "lr";
        private String multiClassClassifierType = "lr";

        private Builder() {
        }

        public Builder setNumClasses(int numClasses) {
            this.numClasses = numClasses;
            return this;
        }

        public Builder setNumComponents(int numComponents) {
            this.numComponents = numComponents;
            return this;
        }

        public Builder setNumFeatures(int numFeatures) {
            this.numFeatures = numFeatures;
            return this;
        }

        public Builder setBinaryClassifierType(String binaryClassifierType) {
            this.binaryClassifierType = binaryClassifierType;
            return this;
        }

        public Builder setMultiClassClassifierType(String multiClassClassifierType) {
            this.multiClassClassifierType = multiClassClassifierType;
            return this;
        }

        public Builder setSupport(List<MultiLabel> support) {
            this.support = support;
            return this;
        }

        public CBM build(){
            CBM CBM = new CBM();
            CBM.numLabels = numClasses;
            CBM.numComponents = numComponents;
            CBM.numFeatures = numFeatures;
            CBM.binaryClassifierType = binaryClassifierType;
            CBM.multiClassClassifierType = multiClassClassifierType;
            CBM.support = support;

            switch (binaryClassifierType){
                case "lr":
                    CBM.binaryClassifiers = new LogisticRegression[numComponents][numClasses];
                    for (int k = 0; k< numComponents; k++) {
                        for (int l=0; l<numClasses; l++) {
                            CBM.binaryClassifiers[k][l] = new LogisticRegression(2,numFeatures);
                        }
                    }
                    break;
                case "boost":
                    CBM.binaryClassifiers = new LKBoost[numComponents][numClasses];
                    for (int k = 0; k< numComponents; k++) {
                        for (int l=0; l<numClasses; l++) {
                            CBM.binaryClassifiers[k][l] = new LKBoost(2);
                        }
                    }
                    break;
                case "elasticnet":
                    CBM.binaryClassifiers = new LogisticRegression[numComponents][numClasses];
                    for (int k = 0; k< numComponents; k++) {
                        for (int l=0; l<numClasses; l++) {
                            CBM.binaryClassifiers[k][l] = new LogisticRegression(2,numFeatures);
                        }
                    }
                    break;
                default:
                    throw new IllegalArgumentException("binaryClassifierType can be lr or boost. Given: "+binaryClassifierType);
            }

            switch (multiClassClassifierType){
                case "lr":
                    CBM.multiClassClassifier = new LogisticRegression(numComponents, numFeatures,true);
                    break;
                case "boost":
                    CBM.multiClassClassifier = new LKBoost(numComponents);
                    break;
                case "elasticnet":
                    CBM.multiClassClassifier = new LogisticRegression(numComponents, numFeatures,true);
                    break;
                default:
                    throw new IllegalArgumentException("multiClassClassifierType can be lr or boost");
            }

            return CBM;
        }
    }

}