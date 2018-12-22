package edu.neu.ccs.pyramid.multilabel_classification.imlgb;

import edu.neu.ccs.pyramid.dataset.MultiLabel;
import edu.neu.ccs.pyramid.multilabel_classification.PluginPredictor;
import org.apache.mahout.math.Vector;

public class CalibratedSetAccPredictor implements PluginPredictor<IMLGradientBoosting> {
    private static final long serialVersionUID = 1L;
    private IMLGradientBoosting imlGradientBoosting;
    private IMLGBLabelIsotonicScaling labelIsotonicScaling;

    public CalibratedSetAccPredictor(IMLGradientBoosting imlGradientBoosting,
                                     IMLGBLabelIsotonicScaling labelIsotonicScaling) {
        this.imlGradientBoosting = imlGradientBoosting;
        this.labelIsotonicScaling = labelIsotonicScaling;
    }

    @Override
    public IMLGradientBoosting getModel() {
        return null;
    }

    @Override
    public MultiLabel predict(Vector vector) {
        if (imlGradientBoosting.getAssignments()==null){
            throw new RuntimeException("acc predictor is used but legal assignments is not specified!");
        }
        double maxProb = Double.NEGATIVE_INFINITY;
        MultiLabel prediction = null;
        double[] uncalibrated = imlGradientBoosting.predictClassProbs(vector);
        double[] calibrated = labelIsotonicScaling.calibratedClassProbs(uncalibrated);
        for (MultiLabel assignment: imlGradientBoosting.getAssignments()){
            double prob = proba(assignment, calibrated);
            if (prob > maxProb){
                maxProb = prob;
                prediction = assignment;
            }
        }
        return prediction;
    }


    private static double proba(MultiLabel multiLabel, double[] marginals){
        double product = 1;
        for (int l=0;l<marginals.length;l++){

            if (multiLabel.matchClass(l)){
                product *= marginals[l];
            } else {
                product *= 1-marginals[l];
            }
        }
        return product;
    }


}
