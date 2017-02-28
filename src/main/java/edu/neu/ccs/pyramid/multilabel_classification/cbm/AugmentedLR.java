package edu.neu.ccs.pyramid.multilabel_classification.cbm;

import edu.neu.ccs.pyramid.util.MathUtil;
import edu.neu.ccs.pyramid.util.Vectors;
import org.apache.mahout.math.Vector;

/**
 * binary logistic regression on augmented vector (x,z)
 * Created by chengli on 2/28/17.
 */
public class AugmentedLR {
    private int numFeatures;
    private int numComponents;
    // size = num features + num components + 1
    private Vector weights;

    private double getWeightForComponent(int k){
        return weights.get(numFeatures+k);
    }

    private double getBias(){
        return weights.get(weights.size()-1);
    }

    private Vector featureWeights(){
        return weights.viewPart(0, numFeatures);
    }

    /**
     * feature part score, including the global bias
     * @param featureVector
     * @return
     */
    private double featureScore(Vector featureVector){
        return Vectors.dot(featureWeights(), featureVector) + getBias();
    }

    /**
     * total score for each (x,z)
     * @param featureVector
     * @return
     */
    private double[] augmentedScores(Vector featureVector){
        double[] scores = new double[numComponents];
        double featureScore = featureScore(featureVector);
        for (int k=0;k<numComponents;k++){
            scores[k] = featureScore + getWeightForComponent(k);
        }
        return scores;
    }

    private double[] logAugmentedProbs(double[] augmentedScores){
        double[] logProbs = new double[numComponents];
        for (int k=0;k<numComponents;k++){
            logProbs[k] = MathUtil.logSigmoid(augmentedScores[k]);
        }
        return logProbs;
    }

    private double[] logAugmentedProbs(Vector featureVector){
        double[] scores = augmentedScores(featureVector);
        return logAugmentedProbs(scores);
    }


}
