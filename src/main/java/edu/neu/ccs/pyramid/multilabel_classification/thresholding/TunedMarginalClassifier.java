package edu.neu.ccs.pyramid.multilabel_classification.thresholding;

import edu.neu.ccs.pyramid.dataset.LabelTranslator;
import edu.neu.ccs.pyramid.dataset.MultiLabel;
import edu.neu.ccs.pyramid.feature.FeatureList;
import edu.neu.ccs.pyramid.multilabel_classification.MultiLabelClassifier;
import org.apache.mahout.math.Vector;

import java.util.Arrays;

/**
 * Created by chengli on 2/28/16.
 */
public class TunedMarginalClassifier implements MultiLabelClassifier{
    MultiLabelClassifier.ClassProbEstimator classProbEstimator;
    double[] thresholds;

    public TunedMarginalClassifier(MultiLabelClassifier.ClassProbEstimator classProbEstimator) {
        this.classProbEstimator = classProbEstimator;
        thresholds = new double[classProbEstimator.getNumClasses()];
    }

    public double[] getThresholds() {
        return thresholds;
    }

    public void setThresholds(double[] thresholds) {
        this.thresholds = thresholds;
    }

    public void setThresholdSameValue(double threshold){
        Arrays.fill(thresholds, threshold);
    }

    @Override
    public int getNumClasses() {
        return classProbEstimator.getNumClasses();
    }

    @Override
    public MultiLabel predict(Vector vector) {
        MultiLabel multiLabel = new MultiLabel();
        int numClasses = classProbEstimator.getNumClasses();
        double[] probs = classProbEstimator.predictClassProbs(vector);
        for (int l=0;l<numClasses;l++){
            if (probs[l] > thresholds[l]){
                multiLabel.addLabel(l);
            }
        }
        return multiLabel;
    }

    @Override
    public FeatureList getFeatureList() {
        return null;
    }

    @Override
    public LabelTranslator getLabelTranslator() {
        return null;
    }
}
