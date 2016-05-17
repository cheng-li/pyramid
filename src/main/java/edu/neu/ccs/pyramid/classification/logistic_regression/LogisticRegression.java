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
    private int numClasses;
    private int numFeatures;
    private Weights weights;
    private boolean featureExtraction = false;
    private FeatureList featureList;
    private LabelTranslator labelTranslator;



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


    public boolean featureExtraction() {
        return featureExtraction;
    }

    public void setFeatureExtraction(boolean featureExtraction) {
        this.featureExtraction = featureExtraction;
    }

    public Weights getWeights() {
        return weights;
    }

    @Override
    public int getNumClasses() {
        return this.numClasses;
    }

    public int getNumFeatures() {
        return numFeatures;
    }

    @Override
    public int predict(Vector vector){
        double[] scores = predictClassScores(vector);
        double maxScore = Double.NEGATIVE_INFINITY;
        int predictedClass = 0;
        for (int k=0;k<this.numClasses;k++){
            double scoreClassK = scores[k];
            if (scoreClassK > maxScore){
                maxScore = scoreClassK;
                predictedClass = k;
            }
        }
        return predictedClass;
    }

    public double predictClassScore(Vector dataPoint, int k){
        double score = 0;
        score += this.weights.getBiasForClass(k);
        score += this.weights.getWeightsWithoutBiasForClass(k).dot(dataPoint);
//        score += Vectors.dot(weights.getWeightsWithoutBiasForClass(k),dataPoint);
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
        double[] probVector = new double[this.numClasses];
        double logDenominator = MathUtil.logSumExp(scoreVector);
        for (int k=0;k<this.numClasses;k++){
            double logNumerator = scoreVector[k];
            double pro = Math.exp(logNumerator-logDenominator);
            probVector[k]=pro;
        }
        return probVector;
    }


    public double[] predictLogClassProbs(Vector vector){
        double[] scoreVector = this.predictClassScores(vector);
        double[] logProbVector = new double[this.numClasses];
        double logDenominator = MathUtil.logSumExp(scoreVector);
        for (int k=0;k<this.numClasses;k++) {
            double logNumerator = scoreVector[k];
            logProbVector[k]=logNumerator-logDenominator;
        }
        return logProbVector;
    }

    double logLikelihood(Vector vector, int k){
        double[] scoreVector = this.predictClassScores(vector);
        double logDenominator = MathUtil.logSumExp(scoreVector);
        double logNumerator = scoreVector[k];
        return logNumerator-logDenominator;
    }

    double klDivergence(Vector vector, double[] targetDistribution){
        double[] logEstimation = predictLogClassProbs(vector);

        return KLDivergence.klGivenPLogQ(targetDistribution,logEstimation);

    }

    double dataSetKLDivergence(DataSet dataSet, double[][] targetDistributions){
        return IntStream.range(0,dataSet.getNumDataPoints()).parallel()
                .mapToDouble(i -> klDivergence(dataSet.getRow(i),targetDistributions[i]))
                .sum();
    }

    public double dataSetKLWeightedDivergence(DataSet dataSet, double[][] targetDistributions, double[] weights) {
        double sum = 0.0;
        for(int n=0; n<dataSet.getNumDataPoints(); n++) {
            sum += weights[n] * klDivergence(dataSet.getRow(n), targetDistributions[n]);
        }
        return sum;
    }

    public double dataSetLogLikelihood(ClfDataSet dataSet){
        int[] labels = dataSet.getLabels();
        return IntStream.range(0,dataSet.getNumDataPoints()).parallel()
                .mapToDouble(i->logLikelihood(dataSet.getRow(i),labels[i]))
                .sum();
    }

    public double dataSetLogLikelihood(ClfDataSet dataSet, int[] labels) {
        return IntStream.range(0,dataSet.getNumDataPoints()).parallel()
                .mapToDouble(i->logLikelihood(dataSet.getRow(i),labels[i]))
                .sum();
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

    void setLabelTranslator(LabelTranslator labelTranslator) {
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
