package edu.neu.ccs.pyramid.classification.dirty_naive_bayes;

import edu.neu.ccs.pyramid.classification.Classifier;
import edu.neu.ccs.pyramid.dataset.LabelTranslator;
import edu.neu.ccs.pyramid.feature.FeatureList;
import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.Vector;

/**
 * Created by chengli on 3/28/15.
 */
public class NaiveBayes implements Classifier{
    int numClasses;
    int numFeatures;
    double[] priors;
    // #class * #feature
    double [][] conditionals;
    double[][] logPositive;
    double[][] logNegative;

    public NaiveBayes(int numClasses, int numFeatures) {
        this.numClasses = numClasses;
        this.numFeatures = numFeatures;
        this.priors = new double[numClasses];
        this.conditionals = new double[numClasses][numFeatures];
        this.logPositive = new double[numClasses][numFeatures];
        this.logNegative = new double[numClasses][numFeatures];
    }

    @Override
    public int predict(Vector vector) {
        int pred = 0;
        double bestScore = Double.NEGATIVE_INFINITY;

        for (int k=0;k<numClasses;k++){
            double score = predictClassScore(vector,k);
            if (score > bestScore){
                bestScore = score;
                pred = k;
            }
        }
        return pred;
    }

    @Override
    public int getNumClasses() {
        return numClasses;
    }

    @Override
    public FeatureList getFeatureList() {
        return null;
    }

    @Override
    public LabelTranslator getLabelTranslator() {
        return null;
    }

    private double predictClassScore(Vector vector, int k){
        double[] conditionForClass = conditionals[k];
        double score = Math.log(priors[k]);
        Vector input;
        if (vector.isDense()){
            input = vector;
        } else {
            input = new DenseVector(vector);
        }
        for (int j=0;j<input.size();j++){
            double value = input.get(j);
            if (value>0){
                score += logPositive[k][j];
            } else {
                score += logNegative[k][j];
            }
        }
        return score;
    }
}
