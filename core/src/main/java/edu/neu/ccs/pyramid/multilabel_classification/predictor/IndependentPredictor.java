package edu.neu.ccs.pyramid.multilabel_classification.predictor;

import edu.neu.ccs.pyramid.calibration.IdentityLabelCalibrator;
import edu.neu.ccs.pyramid.calibration.LabelCalibrator;
import edu.neu.ccs.pyramid.calibration.VectorIsoSetCalibrator;
import edu.neu.ccs.pyramid.dataset.MultiLabel;
import edu.neu.ccs.pyramid.multilabel_classification.MultiLabelClassifier;
import edu.neu.ccs.pyramid.multilabel_classification.PluginPredictor;
import edu.neu.ccs.pyramid.util.Pair;
import org.apache.mahout.math.Vector;

public class IndependentPredictor implements PluginPredictor<MultiLabelClassifier.ClassProbEstimator> {
    private static final long serialVersionUID = 1L;

    MultiLabelClassifier.ClassProbEstimator classifier;
    LabelCalibrator labelCalibrator;

    public IndependentPredictor(ClassProbEstimator classifier, LabelCalibrator labelCalibrator) {
        this.classifier = classifier;
        this.labelCalibrator = labelCalibrator;
    }

    public IndependentPredictor(ClassProbEstimator classifier) {
        this.classifier = classifier;
        this.labelCalibrator = new IdentityLabelCalibrator();
    }

    @Override
    public ClassProbEstimator getModel() {
        return classifier;
    }

    @Override
    public MultiLabel predict(Vector vector) {
        MultiLabel prediction = new MultiLabel();
        double[] uncalibrated = classifier.predictClassProbs(vector);
        double[] calibrated = labelCalibrator.calibratedClassProbs(uncalibrated);
        for (int k=0;k<getNumClasses();k++){
            if (calibrated[k] >= 0.5){
                prediction.addLabel(k);
            }
        }
        return prediction;
    }


    public Pair<MultiLabel, Double> predictWithConfidence(Vector vector){
        MultiLabel prediction = new MultiLabel();
        double[] uncalibrated = classifier.predictClassProbs(vector);
        double[] calibrated = labelCalibrator.calibratedClassProbs(uncalibrated);
        double prod = 1;
        for (int k=0;k<getNumClasses();k++){
            if (calibrated[k] >= 0.5){
                prediction.addLabel(k);
                prod *= calibrated[k];
            } else {
                prod *= 1-calibrated[k];
            }
        }
        return new Pair<>(prediction,prod);
    }
}
