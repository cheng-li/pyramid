package edu.neu.ccs.pyramid.multilabel_classification.crf;

import edu.neu.ccs.pyramid.dataset.LabelTranslator;
import edu.neu.ccs.pyramid.dataset.MultiLabel;
import edu.neu.ccs.pyramid.dataset.MultiLabelClfDataSet;
import edu.neu.ccs.pyramid.feature.FeatureList;
import edu.neu.ccs.pyramid.multilabel_classification.Enumerator;
import edu.neu.ccs.pyramid.multilabel_classification.MultiLabelClassifier;
import edu.neu.ccs.pyramid.util.ArgMax;
import edu.neu.ccs.pyramid.util.MathUtil;
import org.apache.mahout.math.Vector;

import java.io.*;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.stream.IntStream;

import static edu.neu.ccs.pyramid.dataset.DataSetUtil.gatherMultiLabels;

/**
 * Pair-wise Conditional Random Fields
 * See
 * Ghamrawi, Nadia, and Andrew McCallum.
 * "Collective multi-label classification.
 * " Proceedings of the 14th ACM international conference on Information and knowledge management. ACM, 2005.
 * Created by Rainicy on 12/12/15.
 */
public class CMLCRF implements MultiLabelClassifier, Serializable {
    private static final long serialVersionUID = 3L;
    /**
     * Y_1, Y_2,...,Y_L
     */
    private int numClasses;
    /**
     * X feature length
     */
    private int numFeatures;

    private Weights weights;

    private List<MultiLabel> supportCombinations;

    private int numSupports;

    // for each combination, store the total score computed based only on labels
    // Since it doesn't depend on features, it can be re-used by all data points
    private double[] combinationLabelPartScores;

    private boolean considerPair = true;

    private double lossStrength = 1;

    private LabelTranslator labelTranslator;

    private FeatureList featureList;


    public CMLCRF(MultiLabelClfDataSet dataSet) {
        this.numClasses = dataSet.getNumClasses();
        this.numFeatures = dataSet.getNumFeatures();
        this.weights = new Weights(numClasses, numFeatures);

        //todo
        this.supportCombinations = gatherMultiLabels(dataSet);
//        this.supportCombinations = Enumerator.enumerate(dataSet.getNumClasses());
        this.numSupports = supportCombinations.size();

//
//        System.out.println("support combinations: " + supportCombinations);
//        System.out.println("size of support " + this.numSupports);

        this.combinationLabelPartScores = new double[supportCombinations.size()];
        updateCombLabelPartScores();
//        System.out.println("done with updating combined label part scores.");
        this.labelTranslator = dataSet.getLabelTranslator();
        this.featureList = dataSet.getFeatureList();
    }

    public double getLossStrength() {
        return lossStrength;
    }

    public void setLossStrength(double lossStrength) {
        this.lossStrength = lossStrength;
    }

    public boolean considerPair() {
        return considerPair;
    }

    public void setConsiderPair(boolean considerPair) {
        this.considerPair = considerPair;
        updateCombLabelPartScores();
    }


    // for the feature-label pair
    double predictClassScore(Vector vector, int classIndex){
        double score = 0.0;
        score += this.weights.getWeightsWithoutBiasForClass(classIndex).dot(vector);
        score += this.weights.getBiasForClass(classIndex);
        return score;
    }

    double[] predictClassScores(Vector vector){
        double[] scores = new double[numClasses];
        for (int k=0;k<numClasses;k++){
            scores[k] = predictClassScore(vector, k);
        }
        return scores;
    }

    /**
     * get the scores for all possible label combination
     * y and a given feature x.
     * @param vector
     * @return
     */
    public double[] predictCombinationScores(Vector vector){
        double[] classScores = predictClassScores(vector);
        return predictCombinationScores(classScores);
    }

    double[] predictCombinationScores(double[] classScores){
        double[] scores = new double[this.numSupports];
        for (int k=0;k<scores.length;k++){
            scores[k] = predictCombinationScore(k, classScores);
        }
        return scores;
    }


    // todo fix: handle separately
    private double predictCombinationScore(int labelComIndex, double[] classScores){
        MultiLabel label = supportCombinations.get(labelComIndex);
        double score = 0.0;
        for (Integer l: label.getMatchedLabels()){
            score += classScores[l];
        }
        if (considerPair){
            score += combinationLabelPartScores[labelComIndex];
        }

        return score;
    }



    /**
     * get the scores for all possible label combination
     * y and a given feature x.
     * @param vector
     * @return
     */
    public double[] predictLossAugmentedCombinationScores(int trueComIndex, Vector vector, double[][] lossMatrix){
        double[] classScores = predictClassScores(vector);
        return predictLossAugmentedCombinationScores(trueComIndex, classScores, lossMatrix);
    }

    double[] predictLossAugmentedCombinationScores(int trueComIndex, double[] classScores, double[][] lossMatrix){
        double[] scores = new double[this.numSupports];
        for (int k=0;k<scores.length;k++){
            scores[k] = predictLossAugmentedCombinationScore(trueComIndex, k, classScores, lossMatrix);
        }
        return scores;
    }


    /**
     *
     * @param predictComIndex
     * @param classScores
     * @param lossMatrix numSupport* numSupport. e.g., 1-F1 score; format: [true comb index][predict com index]
     * @return
     */
    private double predictLossAugmentedCombinationScore(int trueComIndex, int predictComIndex,  double[] classScores, double[][] lossMatrix){
        double original = predictCombinationScore(predictComIndex, classScores);
        return original + lossStrength*lossMatrix[trueComIndex][predictComIndex];
    }


    /**
     * the part of score which depends only on labels
     * for each label pair, exactly one feature function returns 1
     * @return
     */
    double computeLabelPartScore(int labelComIndex){
        MultiLabel label = supportCombinations.get(labelComIndex);
        double score = 0;
        int pos = this.weights.getNumWeightsForFeatures();
        boolean[] matches = new boolean[numClasses];
        for (int match: label.getMatchedLabels()){
            matches[match] = true;
        }
        for (int l1=0; l1<numClasses; l1++) {
            for (int l2=l1+1; l2<numClasses; l2++) {
                if (!matches[l1] && !matches[l2]) {
                    score += this.weights.getWeightForIndex(pos);
                } else if (matches[l1] && !matches[l2]) {
                    score += this.weights.getWeightForIndex(pos + 1);
                } else if (!matches[l1] && matches[l2]) {
                    score += this.weights.getWeightForIndex(pos + 2);
                } else {
                    score += this.weights.getWeightForIndex(pos + 3);
                }
                pos += 4;
            }
        }

        return score;
    }

    void updateCombLabelPartScores(){
        IntStream.range(0, supportCombinations.size()).parallel()
                .forEach(c -> combinationLabelPartScores[c]=computeLabelPartScore(c));
    }


    public double[] predictCombinationProbs(Vector vector){
        double[] combinationScores = predictCombinationScores(vector);
        return predictCombinationProbs(combinationScores);
    }

    public double[] predictCombinationProbs(double[] combinationScores){
        return MathUtil.softmax(combinationScores);
    }


    public double[] predictLogCombinationProbs(Vector vector){
        double[] scoreVector = this.predictCombinationScores(vector);
        double[] logProbVector = new double[this.numSupports];
        double logDenominator = MathUtil.logSumExp(scoreVector);
        for (int k=0;k<this.numSupports;k++) {
            double logNumerator = scoreVector[k];
            logProbVector[k]=logNumerator-logDenominator;
        }
        return logProbVector;
    }

    /**
     * marginal probabilities
     * @param assignmentProbs
     * @return
     */
    public double[] calClassProbs(double[] assignmentProbs){
        double[] classProbs = new double[numClasses];
        for (int a=0;a< numSupports;a++){
            MultiLabel assignment = supportCombinations.get(a);
            double prob = assignmentProbs[a];
            for (Integer label:assignment.getMatchedLabels()){
                classProbs[label] += prob;
            }
        }
        return classProbs;
    }

    public double[] predictClassProbs(Vector vector){
        double[] combProbs = predictCombinationProbs(vector);
        return calClassProbs(combProbs);
    }

    @Override
    public int getNumClasses() {
        return numClasses;
    }

    public int getNumSupports() {
        return numSupports;
    }

    public int getNumFeatures() {
        return numFeatures;
    }

    public Weights getWeights() {
        return weights;
    }

    public List<MultiLabel> getSupportCombinations() {
        return supportCombinations;
    }


    @Override
    public MultiLabel predict(Vector vector) {
//        return predictByArgmax(vector);
        double[] scores = predictCombinationScores(vector);
        int predictedCombination = ArgMax.argMax(scores);

        return this.supportCombinations.get(predictedCombination);
    }

    public MultiLabel predictByArgmax(Vector vector) {
        double[] scores = predictClassScores(vector);
        MultiLabel label = new MultiLabel();
        for (int l=0; l<scores.length; l++) {
            if (scores[l] > 0) {
                label.addLabel(l);
            }
        }
        return label;
    }

    @Override
    public FeatureList getFeatureList() {
        return featureList;
    }

    @Override
    public LabelTranslator getLabelTranslator() {
        return labelTranslator;
    }


    // TODO
    @Override
    public String toString() {
        return getWeights().toString();
    }

    public static CMLCRF deserialize(File file) throws Exception {
        try (
                FileInputStream fileInputStream = new FileInputStream(file);
                BufferedInputStream bufferedInputStream = new BufferedInputStream(fileInputStream);
                ObjectInputStream objectInputStream = new ObjectInputStream(bufferedInputStream);
        ){
            CMLCRF cmlcrf = (CMLCRF) objectInputStream.readObject();
            return cmlcrf;
        }
    }

    public static CMLCRF deserialize(String file) throws Exception {
        File file1 = new File(file);
        return deserialize(file1);
    }

    @Override
    public void serialize(File file) throws Exception {
        File parent = file.getParentFile();
        if (!parent.exists()) {
            parent.mkdir();
        }
        try (
                FileOutputStream fileOutputStream = new FileOutputStream(file);
                BufferedOutputStream bufferedOutputStream = new BufferedOutputStream(fileOutputStream);
                ObjectOutputStream objectOutputStream = new ObjectOutputStream(bufferedOutputStream);
        ){
            objectOutputStream.writeObject(this);
        }
    }

    @Override
    public void serialize(String file) throws Exception {
        File file1 = new File(file);
        serialize(file1);
    }

}
