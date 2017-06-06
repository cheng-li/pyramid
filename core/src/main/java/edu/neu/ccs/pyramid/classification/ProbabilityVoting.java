package edu.neu.ccs.pyramid.classification;


import edu.neu.ccs.pyramid.dataset.LabelTranslator;
import edu.neu.ccs.pyramid.feature.FeatureList;
import org.apache.mahout.math.Vector;

import java.util.ArrayList;
import java.util.List;

/**
 * Created by chengli on 9/9/14.
 */
public class ProbabilityVoting implements Classifier.ProbabilityEstimator {
    private int numClasses;
    private List<ProbabilityEstimator> estimatorList;
    private FeatureList featureList;
    private LabelTranslator labelTranslator;

    public ProbabilityVoting(int numClasses) {
        this.numClasses = numClasses;
        this.estimatorList = new ArrayList<>();
    }

    public void add(ProbabilityEstimator estimator){
        if (estimator.getNumClasses()!=this.numClasses){
            throw new IllegalArgumentException("illegal number of classes");
        }
        this.estimatorList.add(estimator);
    }

    public int predict(Vector vector){
        int numEstimators = this.estimatorList.size();
        double[] averageProbs = new double[this.numClasses];
        for (ProbabilityEstimator estimator: estimatorList){
            double[] probs = estimator.predictClassProbs(vector);
            for (int k=0;k<this.numClasses;k++){
                averageProbs[k] += probs[k];
            }
        }
        //normalization is unnecessary if the goal is just to predict
        for (int k=0;k<this.numClasses;k++){
            averageProbs[k] /= numEstimators;
        }
        int pred = 0;
        double maxProb = averageProbs[0];
        for (int k=0;k<this.numClasses;k++){
            if (averageProbs[k]> maxProb){
                maxProb = averageProbs[k];
                pred = k;
            }
        }
        return pred;
    }

    public int size(){
        return this.estimatorList.size();
    }

    public ProbabilityEstimator getProbEstimator(int i){
        return this.estimatorList.get(i);
    }

    @Override
    public int getNumClasses() {
        return this.numClasses;
    }

    @Override
    public double[] predictClassProbs(Vector vector) {
        int numEstimators = this.estimatorList.size();
        double[] averageProbs = new double[this.numClasses];
        for (ProbabilityEstimator estimator: estimatorList){
            double[] probs = estimator.predictClassProbs(vector);
            for (int k=0;k<this.numClasses;k++){
                averageProbs[k] += probs[k];
            }
        }
        //normalization is unnecessary if the goal is just to predict
        for (int k=0;k<this.numClasses;k++){
            averageProbs[k] /= numEstimators;
        }
        return averageProbs;
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
