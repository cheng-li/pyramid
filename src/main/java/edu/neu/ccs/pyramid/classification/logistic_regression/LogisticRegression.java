package edu.neu.ccs.pyramid.classification.logistic_regression;

import edu.neu.ccs.pyramid.classification.ProbabilityEstimator;
import edu.neu.ccs.pyramid.util.MathUtil;
import org.apache.mahout.math.Vector;

import java.io.*;

/**
 * Created by chengli on 11/28/14.
 */
public class LogisticRegression implements ProbabilityEstimator {
    private static final long serialVersionUID = 1L;
    private int numClasses;
    private int numFeatures;
    private Weights weights;




    public LogisticRegression(int numClasses, int numFeatures) {
        this.numClasses = numClasses;
        this.numFeatures = numFeatures;
        this.weights = new Weights(numClasses, numFeatures);
    }

    public Weights getWeights() {
        return weights;
    }

    @Override
    public int getNumClasses() {
        return this.numClasses;
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
            double logNominator = scoreVector[k];
            double pro = Math.exp(logNominator-logDenominator);
            probVector[k]=pro;
        }
        return probVector;
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


}
