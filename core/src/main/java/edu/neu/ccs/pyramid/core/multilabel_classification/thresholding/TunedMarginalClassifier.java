package edu.neu.ccs.pyramid.core.multilabel_classification.thresholding;

import edu.neu.ccs.pyramid.core.dataset.LabelTranslator;
import edu.neu.ccs.pyramid.core.dataset.MultiLabel;
import edu.neu.ccs.pyramid.core.feature.FeatureList;
import edu.neu.ccs.pyramid.core.multilabel_classification.MultiLabelClassifier;
import org.apache.mahout.math.Vector;

import java.util.Arrays;

/**
 * Created by chengli on 2/28/16.
 */
public class TunedMarginalClassifier implements MultiLabelClassifier{
    private static final long serialVersionUID = 1L;

    private MultiLabelClassifier.ClassProbEstimator classProbEstimator;
    private double[] thresholds;

    public TunedMarginalClassifier(MultiLabelClassifier.ClassProbEstimator classProbEstimator) {
        this.classProbEstimator = classProbEstimator;
        thresholds = new double[classProbEstimator.getNumClasses()];
    }

    public TunedMarginalClassifier(ClassProbEstimator classProbEstimator, double[] thresholds) {
        this.classProbEstimator = classProbEstimator;
        this.thresholds = thresholds;
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
