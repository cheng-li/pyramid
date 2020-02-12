package edu.neu.ccs.pyramid.classification.logistic_regression;

import edu.neu.ccs.pyramid.classification.Classifier;
import edu.neu.ccs.pyramid.dataset.ClfDataSet;
import edu.neu.ccs.pyramid.dataset.DataSet;
import edu.neu.ccs.pyramid.dataset.LabelTranslator;
import edu.neu.ccs.pyramid.eval.KLDivergence;
import edu.neu.ccs.pyramid.feature.FeatureList;
import edu.neu.ccs.pyramid.util.MathUtil;
import edu.neu.ccs.pyramid.util.Vectors;
import org.apache.mahout.math.Vector;

import java.io.*;
import java.util.stream.IntStream;

/**
 * Created by chengli on 11/28/14.
 */
public class LogisticRegression implements Classifier.ProbabilityEstimator, Classifier.ScoreEstimator {
    private static final long serialVersionUID = 2L;

    private boolean symmetry = false; // whether the binary LR is symmetric; numClasses==1 is required
    private int numClasses;
    private int numFeatures;
    private Weights weights; // TODO: add sparse option to Weights
    private FeatureList featureList;
    private LabelTranslator labelTranslator;

    /**
     * Logistic Regression Constructor when symmetry is an option.
     * @param numClasses
     * @param numFeatures
     * @param random
     * @param symmetry
     */
    public LogisticRegression(int numClasses, int numFeatures, boolean random, boolean symmetry) {
        if (symmetry) {
            assert numClasses == 2 : " Binary Class is required for Symmetric (Binary) LogisticRegression";
        }

        this.symmetry = symmetry;
        this.numClasses = numClasses;
        this.numFeatures = numFeatures;
        // if symmetry, only creates one class weight vector.
        this.weights = new Weights(symmetry ? 1 : numClasses, numFeatures, random);
    }

    public LogisticRegression(int numClasses, int numFeatures, boolean random) {
        this.numClasses = numClasses;
        this.numFeatures = numFeatures;
        this.weights = new Weights(numClasses, numFeatures, random);
    }

    public LogisticRegression(int numClasses, int numFeatures) {
        this.numClasses = numClasses;
        this.numFeatures = numFeatures;
        this.weights = new Weights(numClasses, numFeatures);
    }

    public LogisticRegression(int numClasses, int numFeatures, Vector weightVector) {
        this.numClasses = numClasses;
        this.numFeatures = numFeatures;
        this.weights = new Weights(numClasses, numFeatures, weightVector);
    }

    /**
     * a logistic regression that gives p(y=k|x) = prior p(y=k)
     * @param numClasses
     * @param numFeatures
     * @param priorProbabilities
     */
    public LogisticRegression(int numClasses, int numFeatures, double[] priorProbabilities) {
        this.numClasses = numClasses;
        this.numFeatures = numFeatures;
        this.weights = new Weights(numClasses, numFeatures);
        double[] scores = MathUtil.inverseSoftMax(priorProbabilities);
        for (int l=0;l<numClasses;l++){
            weights.setBiasForClass(scores[l],l);
        }
    }


    public Weights getWeights() {
        return weights;
    }

    @Override
    public int getNumClasses() {
        return this.numClasses;
    }

    public boolean isSymmetry() {
        return this.symmetry;
    }

    public int getNumFeatures() {
        return numFeatures;
    }

    @Override
    public int predict(Vector vector){
        double[] scores = predictClassScores(vector);
        int predictedClass = 0;
        double maxScore = Double.NEGATIVE_INFINITY;
        for (int k=0;k<scores.length;k++){
            double scoreClassK = scores[k];
            if (scoreClassK > maxScore){
                maxScore = scoreClassK;
                predictedClass = k;
            }
        }
        return predictedClass;
    }

    /**
     * the linear scores for given class k
     * when symmetry = true; k=0 for positive score; and the negative score=0;
     * when symmetry = false; k stands for the class number
     * @param dataPoint
     * @param k
     * @return
     */
    public double predictClassScore(Vector dataPoint, int k){
        if (this.symmetry) {
            return k == 0 ? 0 : this.weights.getBiasForClass(0) +
                    Vectors.dot(weights.getWeightsWithoutBiasForClass(0), dataPoint);
        }

        double score = 0;
        score += this.weights.getBiasForClass(k);
        // use our own implementation
//        score += this.weights.getWeightsWithoutBiasForClass(k).dot(dataPoint);
        score += Vectors.dot(weights.getWeightsWithoutBiasForClass(k),dataPoint); // TODO: switch the weights to sparse
        return score;
    }

    public double[] predictClassScores(Vector dataPoint){
        double[] scores = new double[this.numClasses];
        for (int k=0;k<this.numClasses;k++){
            scores[k] = predictClassScore(dataPoint, k);
        }
        return scores;
    }

    @Override
    public double[] predictClassProbs(Vector vector){
        double[] scoreVector = this.predictClassScores(vector);
        double[] probVector = new double[scoreVector.length];
        double logDenominator = MathUtil.logSumExp(scoreVector);
        for (int k=0;k<scoreVector.length;k++){
            double logNumerator = scoreVector[k];
            double pro = Math.exp(logNumerator-logDenominator);
            probVector[k]=pro;
        }
        return probVector;
    }

    /**
     * when the scoreVector is given, ignoring the predictClassScores step.
     * @param scoreVector
     * @return
     */
    public double[] predictClassProbs(double[] scoreVector) {
        double[] probVector = new double[scoreVector.length];
        double logDenominator = MathUtil.logSumExp(scoreVector);
        for (int k=0;k<scoreVector.length;k++){
            double logNumerator = scoreVector[k];
            double pro = Math.exp(logNumerator-logDenominator);
            probVector[k]=pro;
        }
        return probVector;
    }


    public double[] predictLogClassProbs(Vector vector){
        double[] scoreVector = this.predictClassScores(vector);
        double[] logProbVector = new double[scoreVector.length];
        double logDenominator = MathUtil.logSumExp(scoreVector);
        for (int k=0;k<scoreVector.length;k++) {
            double logNumerator = scoreVector[k];
            logProbVector[k]=logNumerator-logDenominator;
        }
        return logProbVector;
    }

    /**
     * logLikelihood for each datapoint.
     * @param vector
     * @param targets
     * @return
     */
    double logLikelihood(Vector vector, double[] targets) {
        double[] scoreVector = this.predictClassScores(vector);
        double logDenominator = MathUtil.logSumExp(scoreVector);
        double logNumberator = 0.0;
        for (int k=0; k<scoreVector.length; k++) {
            logNumberator += targets[k] * scoreVector[k];
        }
        return logNumberator - logDenominator;
    }

    /**
     * logLikelihood for each datapoint.
     * @param vector
     * @param targets
     * @return
     */
    double logLikelihood(Vector vector, double[] targets, double weight) {
        double[] scoreVector = this.predictClassScores(vector);
        double logDenominator = MathUtil.logSumExp(scoreVector);
        double logNumberator = 0.0;
        for (int k=0; k<scoreVector.length; k++) {
            logNumberator += targets[k] * scoreVector[k];
        }
        return weight*(logNumberator - logDenominator);
    }


    public double dataSetLogLikelihood(DataSet dataSet, double[][] targets) {
        return IntStream.range(0, dataSet.getNumDataPoints()).parallel()
                .mapToDouble(i -> logLikelihood(dataSet.getRow(i), targets[i]))
                .sum();
    }

    public double dataSetLogLikelihood(DataSet dataSet, double[][] targets, double[] weights) {
        return IntStream.range(0, dataSet.getNumDataPoints()).parallel()
                .mapToDouble(i -> logLikelihood(dataSet.getRow(i), targets[i], weights[i]))
                .sum();
    }

    /**
     * when the targets are exclusive, e.g. multi-class, targets<numClasses
     * @param dataSet
     * @param targets
     * @return
     */
    public double dataSetLogLikelihood(DataSet dataSet, int[] targets) {
        return IntStream.range(0, dataSet.getNumDataPoints()).parallel()
                .mapToDouble(i -> logLikelihood(dataSet.getRow(i), targets[i], 1))
                .sum();
    }

    /**
     * when the targets are exclusive, e.g. multi-class
     * @param dataSet
     * @param targets
     * @param weights
     * @return
     */
    public double dataSetLogLikelihood(DataSet dataSet, int[] targets, double[] weights) {
        return IntStream.range(0, dataSet.getNumDataPoints()).parallel()
                .mapToDouble(i -> logLikelihood(dataSet.getRow(i), targets[i], weights[i]))
                .sum();
    }

    /**
     * logLikelihood for each datapoint, while target is exclusive, e.g. multi-class
     * @param vector
     * @param target
     * @return
     */
    double logLikelihood(Vector vector, int target, double weight) {
        int[] targets = symmetry ? new int[2] : new int[numClasses];
        targets[target] = 1;
        double[] scoreVector = this.predictClassScores(vector);
        double logDenominator = MathUtil.logSumExp(scoreVector);
        double logNumberator = 0.0;
        for (int k=0; k<scoreVector.length; k++) {
            logNumberator += targets[k] * scoreVector[k];
        }
        return weight*(logNumberator - logDenominator);
    }


    public void truncateByThreshold(double threshold){
        weights.truncateByThreshold(threshold);
    }


    public static LogisticRegression deserialize(File file) throws Exception{
        try(
                FileInputStream fileInputStream = new FileInputStream(file);
                BufferedInputStream bufferedInputStream = new BufferedInputStream(fileInputStream);
                ObjectInputStream objectInputStream = new ObjectInputStream(bufferedInputStream);
        ){
            return (LogisticRegression)objectInputStream.readObject();
        }
    }

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

    @Override
    public String toString() {
        final StringBuilder sb = new StringBuilder("LogisticRegression{");
        sb.append("numClasses=").append(numClasses);
        sb.append(", numFeatures=").append(numFeatures);
        sb.append(", weights=").append(weights);
        sb.append('}');
        return sb.toString();
    }
}
