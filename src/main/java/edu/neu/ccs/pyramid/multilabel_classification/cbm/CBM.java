package edu.neu.ccs.pyramid.multilabel_classification.cbm;

import edu.neu.ccs.pyramid.classification.lkboost.LKBoost;
import edu.neu.ccs.pyramid.classification.logistic_regression.LogisticRegression;
import edu.neu.ccs.pyramid.dataset.LabelTranslator;
import edu.neu.ccs.pyramid.dataset.MultiLabel;
import edu.neu.ccs.pyramid.feature.FeatureList;
import edu.neu.ccs.pyramid.multilabel_classification.MultiLabelClassifier;
import edu.neu.ccs.pyramid.classification.Classifier.ProbabilityEstimator;
import edu.neu.ccs.pyramid.util.BernoulliDistribution;
import edu.neu.ccs.pyramid.util.MathUtil;
import edu.neu.ccs.pyramid.util.Pair;
import org.apache.commons.math3.distribution.EnumeratedIntegerDistribution;
import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.RandomAccessSparseVector;
import org.apache.mahout.math.Vector;

import java.io.*;
import java.util.*;
import java.util.stream.Collectors;
import java.util.stream.IntStream;


/**
 * Conditional Bernoulli Mixtures for Multi-label Classification.
 Cheng Li, Bingyu Wang, Virgil Pavlu, and Javed Aslam.
 In Proceedings of the 33rd International Conference on Machine Learning (ICML), 2016.
 * Created by Rainicy on 10/23/15.
 */
public class CBM implements MultiLabelClassifier.ClassProbEstimator, Serializable {
    private static final long serialVersionUID = 1L;
    int numLabels;
    int numClusters;
    int numFeatures;
    int numSample = 100;
    boolean allowEmpty = false;

    String predictMode = "dynamic";

    // parameters
    // format: [numClusters][numLabels]
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

    /**
     * return the log[p(y_n | z_n=k, x_n; w_k)] by all k from 1 to K.
     * @param X
     * @param Y
     * @return
     */
    double[] clusterConditionalLogProbArr(Vector X, Vector Y) {
        double[] probArr = new double[numClusters];

        for (int k=0; k<numClusters; k++) {
            probArr[k] = clusterConditionalLogProb(X, Y, k);
        }

        return probArr;
    }

    /**
     * return one value for log [p(y_n | z_n=k, x_n; w_k)] by given k;
     * @param X
     * @param Y
     * @param k
     * @return
     */
    private double clusterConditionalLogProb(Vector X, Vector Y, int k) {
        double logProbResult = 0.0;
        for (int l=0; l< binaryClassifiers[k].length; l++) {
            double[] logProbs = binaryClassifiers[k][l].predictLogClassProbs(X);
            if (Y.get(l) == 1.0) {
                logProbResult += logProbs[1];
            } else {
                logProbResult += logProbs[0];
            }
        }
        return logProbResult;
    }


    /**
     * for single assignment, compute log assignment probability
     * @param vector
     * @param assignment
     * @return
     */
    private double predictLogAssignmentProb(Vector vector, MultiLabel assignment){
        double[] logProportions = multiClassClassifier.predictLogClassProbs(vector);
        double[][][] logClassProbs = new double[numClusters][numLabels][2];

        for (int k=0;k<numClusters;k++){
            for (int l=0;l<numLabels;l++){
                logClassProbs[k][l] = binaryClassifiers[k][l].predictLogClassProbs(vector);
            }
        }
        return predictLogAssignmentProb(assignment,logProportions,logClassProbs);
    }

    public double predictAssignmentProb(Vector vector, MultiLabel assignment){
        return Math.exp(predictLogAssignmentProb(vector,assignment));
    }


    /**
     * for batch jobs, use this to save computation
     * @param vector
     * @param assignments
     * @return
     */
    public List<Double> predictLogAssignmentProbs(Vector vector, List<MultiLabel> assignments){
        double[] logProportions = multiClassClassifier.predictLogClassProbs(vector);
        double[][][] logClassProbs = new double[numClusters][numLabels][2];

        for (int k=0;k<numClusters;k++){
            for (int l=0;l<numLabels;l++){
                logClassProbs[k][l] = binaryClassifiers[k][l].predictLogClassProbs(vector);
            }
        }

        List<Double> probs = new ArrayList<>();
        for (MultiLabel multiLabel: assignments){
            probs.add(predictLogAssignmentProb(multiLabel,logProportions,logClassProbs));
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

    /**
     * according to dynamic results, return the top label combinations and the corresponding probs.
     * @param vector
     * @return
     */
    public Pair<List<MultiLabel>, List<Double>> predictDynamicProbs(Vector vector) {
        CBMPredictor cbmPredictor = new CBMPredictor(vector, multiClassClassifier, binaryClassifiers, numClusters, numLabels);
        cbmPredictor.setAllowEmpty(allowEmpty);
        return cbmPredictor.getDynamicProbs();
    }


    private double predictLogAssignmentProb(MultiLabel assignment,double[] logProportions,double[][][] logClassProbs){
        double[] logProbs = new double[numClusters];
        for (int k=0;k<numClusters;k++){
            double sum = 0;
            sum += logProportions[k];
            for (int l=0;l<numLabels;l++){
                if (assignment.matchClass(l)){
                    sum += logClassProbs[k][l][1];
                } else {
                    sum += logClassProbs[k][l][0];
                }
            }
            logProbs[k] = sum;
        }
        return MathUtil.logSumExp(logProbs);
    }


    public MultiLabel predict(Vector vector) {

        // new a BMMPredictor
        CBMPredictor CBMPredictor = new CBMPredictor(vector, multiClassClassifier, binaryClassifiers, numClusters, numLabels);
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
        double[] proportions = getMultiClassClassifier().predictClassProbs(vector);
        double[][] probabilities = new double[numClusters][numLabels];
        for (int k=0;k<numClusters;k++){
            for (int l=0;l<numLabels;l++){
                probabilities[k][l]=getBinaryClassifiers()[k][l].predictClassProb(vector,1);
            }
        }
        double[] probs = new double[numLabels];
        for (int l=0;l<numLabels;l++){
            double sum = 0;
            for (int k=0;k<numClusters;k++){
                sum += proportions[k]*probabilities[k][l];
            }
            probs[l] = sum;
        }
        return probs;
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
        final StringBuilder sb = new StringBuilder("BMM{\n");
        sb.append("numLabels=").append(numLabels).append("\n");
        sb.append("numClusters=").append(numClusters).append("\n");
        for (int k=0;k<numClusters;k++){
            sb.append("cluster ").append(k).append(":\n");
            sb.append("proportion = ").append(mixtureCoefficients[k]).append("\n");
        }

        sb.append("clustering component = \n");
        sb.append(multiClassClassifier);
        sb.append("prediction components = \n");
        for (int k=0;k<numClusters;k++){
            for (int l=0;l<numLabels;l++){
                sb.append("cluster ").append(k).append(" class ").append(l).append("\n");
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
     * @param vector
     * @param numSamples
     * @return
     */
    public List<MultiLabel> samples(Vector vector, int numSamples){
        List<MultiLabel> list = new ArrayList<>();
        double[] proportions = multiClassClassifier.predictClassProbs(vector);
        double[][][] logClassProbs = new double[numClusters][numLabels][2];

        for (int k=0;k<numClusters;k++){
            for (int l=0;l<numLabels;l++){
                logClassProbs[k][l] = binaryClassifiers[k][l].predictLogClassProbs(vector);
            }
        }

        int[] clusters = IntStream.range(0, numClusters).toArray();
        EnumeratedIntegerDistribution enumeratedIntegerDistribution = new EnumeratedIntegerDistribution(clusters, proportions);

        for (int s=0; s<numSamples; s++) {
            int k = enumeratedIntegerDistribution.sample();
            Vector candidateY = new DenseVector(numLabels);

            for (int l=0; l<numLabels; l++) {
                BernoulliDistribution bernoulliDistribution = new BernoulliDistribution(Math.exp(logClassProbs[k][l][1]));
                candidateY.set(l, bernoulliDistribution.sample());
            }
            MultiLabel multiLabel = new MultiLabel(candidateY);
            list.add(multiLabel);
        }
        return list;

    }

    /**
     * sample until the total probability mass of unique subsets exceeds the threshold
     * @param vector
     * @param probMassThreshold
     * @return
     */
    public Pair<List<MultiLabel>, List<Double>> samples(Vector vector, double probMassThreshold){
        int maxNumSamples = 1000;
        List<MultiLabel> multiLabels = new ArrayList<>();
        List<Double> probs = new ArrayList<>();
        double[] logProportions = multiClassClassifier.predictLogClassProbs(vector);
        double[] proportions = Arrays.stream(logProportions).map(Math::exp).toArray();
        double[][][] logClassProbs = new double[numClusters][numLabels][2];
        double[][] classProbs = new double[numClusters][numLabels];
        for (int k=0;k<numClusters;k++){
            for (int l=0;l<numLabels;l++){
                logClassProbs[k][l] = binaryClassifiers[k][l].predictLogClassProbs(vector);
                classProbs[k][l] = Math.exp(logClassProbs[k][l][1]);
            }
        }

        int[] clusters = IntStream.range(0, numClusters).toArray();
        EnumeratedIntegerDistribution enumeratedIntegerDistribution = new EnumeratedIntegerDistribution(clusters, proportions);

        double mass = 0;
        Set<MultiLabel> unique = new HashSet<>();
        int numSamples = 0;
        while (true) {
            numSamples += 1;
//            System.out.println("samples = "+numSamples);
            int k = enumeratedIntegerDistribution.sample();
            Vector candidateY = new DenseVector(numLabels);

            for (int l=0; l<numLabels; l++) {
                BernoulliDistribution bernoulliDistribution = new BernoulliDistribution(classProbs[k][l]);
                candidateY.set(l, bernoulliDistribution.sample());
            }
            MultiLabel multiLabel = new MultiLabel(candidateY);

            if (!unique.contains(multiLabel)){
//                System.out.println("new one");
                multiLabels.add(multiLabel);
                double p = Math.exp(predictLogAssignmentProb(multiLabel,logProportions, logClassProbs));
                probs.add(p);
                mass += p;
                unique.add(multiLabel);
//                System.out.println("mass = "+mass);
            }

            if (mass>probMassThreshold || numSamples == maxNumSamples){
                break;
            }
        }
        return new Pair<>(multiLabels, probs);

    }

    /**
     * only consider non empty sets
     * using the conditional probability p(y|y non empty, x)=p(y, y non empty|x) / p(y non empty |x)
     * sample until the total probability mass of unique subsets exceeds the threshold
     * @param vector
     * @param probMassThreshold
     * @return
     */
    public Pair<List<MultiLabel>, List<Double>> sampleNonEmptySets(Vector vector, double probMassThreshold){
        int maxNumSamples = 1000;
        List<MultiLabel> multiLabels = new ArrayList<>();
        List<Double> probs = new ArrayList<>();
        double[] logProportions = multiClassClassifier.predictLogClassProbs(vector);
        double[] proportions = Arrays.stream(logProportions).map(Math::exp).toArray();
        double[][][] logClassProbs = new double[numClusters][numLabels][2];
        double[][] classProbs = new double[numClusters][numLabels];
        for (int k=0;k<numClusters;k++){
            for (int l=0;l<numLabels;l++){
                logClassProbs[k][l] = binaryClassifiers[k][l].predictLogClassProbs(vector);
                classProbs[k][l] = Math.exp(logClassProbs[k][l][1]);
            }
        }

        int[] clusters = IntStream.range(0, numClusters).toArray();
        EnumeratedIntegerDistribution enumeratedIntegerDistribution = new EnumeratedIntegerDistribution(clusters, proportions);

        MultiLabel emptySet = new MultiLabel();
        double emptyProb = Math.exp(predictLogAssignmentProb(emptySet,logProportions, logClassProbs));
        double nonEmptyProb = 1-emptyProb;

        double mass = 0;
        Set<MultiLabel> unique = new HashSet<>();
        int numSamples = 0;
        while (true) {
            numSamples += 1;
            int k = enumeratedIntegerDistribution.sample();
            Vector candidateY = new DenseVector(numLabels);

            for (int l=0; l<numLabels; l++) {
                BernoulliDistribution bernoulliDistribution = new BernoulliDistribution(classProbs[k][l]);
                candidateY.set(l, bernoulliDistribution.sample());
            }
            MultiLabel multiLabel = new MultiLabel(candidateY);

            if (multiLabel.getNumMatchedLabels()>0 && !unique.contains(multiLabel)){
                multiLabels.add(multiLabel);
                double p = Math.exp(predictLogAssignmentProb(multiLabel,logProportions, logClassProbs));
                double conditionalP = p/nonEmptyProb;
                probs.add(conditionalP);
                mass += conditionalP;
                unique.add(multiLabel);
            }

            if (mass>probMassThreshold || numSamples == maxNumSamples){
                break;
            }
        }
        return new Pair<>(multiLabels, probs);

    }



    public ProbabilityEstimator[][] getBinaryClassifiers() {
        return binaryClassifiers;
    }

    public ProbabilityEstimator getMultiClassClassifier() {
        return multiClassClassifier;
    }

    public int getNumClusters() {
        return numClusters;
    }

    public static Builder getBuilder(){
        return new Builder();
    }


    public static class Builder {
        private int numClasses;
        private int numClusters;
        private int numFeatures;
        private String binaryClassifierType= "lr";
        private String multiClassClassifierType = "lr";

        private Builder() {
        }

        public Builder setNumClasses(int numClasses) {
            this.numClasses = numClasses;
            return this;
        }

        public Builder setNumClusters(int numClusters) {
            this.numClusters = numClusters;
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

        public CBM build(){
            CBM CBM = new CBM();
            CBM.numLabels = numClasses;
            CBM.numClusters = numClusters;
            CBM.numFeatures = numFeatures;
            CBM.binaryClassifierType = binaryClassifierType;
            CBM.multiClassClassifierType = multiClassClassifierType;

            switch (binaryClassifierType){
                case "lr":
                    CBM.binaryClassifiers = new LogisticRegression[numClusters][numClasses];
                    for (int k=0; k<numClusters; k++) {
                        for (int l=0; l<numClasses; l++) {
                            CBM.binaryClassifiers[k][l] = new LogisticRegression(2,numFeatures);
                        }
                    }
                    break;
                case "boost":
                    CBM.binaryClassifiers = new LKBoost[numClusters][numClasses];
                    for (int k=0; k<numClusters; k++) {
                        for (int l=0; l<numClasses; l++) {
                            CBM.binaryClassifiers[k][l] = new LKBoost(2);
                        }
                    }
                    break;
                case "elasticnet":
                    CBM.binaryClassifiers = new LogisticRegression[numClusters][numClasses];
                    for (int k=0; k<numClusters; k++) {
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
                    CBM.multiClassClassifier = new LogisticRegression(numClusters, numFeatures,true);
                    break;
                case "boost":
                    CBM.multiClassClassifier = new LKBoost(numClusters);
                    break;
                case "elasticnet":
                    CBM.multiClassClassifier = new LogisticRegression(numClusters, numFeatures,true);
                    break;
                default:
                    throw new IllegalArgumentException("multiClassClassifierType can be lr or boost");
            }

            return CBM;
        }
    }
}
