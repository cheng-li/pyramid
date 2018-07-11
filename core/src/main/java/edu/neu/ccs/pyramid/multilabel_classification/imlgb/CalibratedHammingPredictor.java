package edu.neu.ccs.pyramid.multilabel_classification.imlgb;

import edu.neu.ccs.pyramid.dataset.MultiLabel;
import edu.neu.ccs.pyramid.multilabel_classification.PluginPredictor;
import org.apache.mahout.math.Vector;

public class CalibratedHammingPredictor implements PluginPredictor<IMLGradientBoosting> {
    private static final long serialVersionUID = 1L;
    private IMLGradientBoosting imlGradientBoosting;
    private IMLGBLabelIsotonicScaling labelIsotonicScaling;


    public CalibratedHammingPredictor(IMLGradientBoosting imlGradientBoosting, IMLGBLabelIsotonicScaling labelIsotonicScaling) {
        this.imlGradientBoosting = imlGradientBoosting;
        this.labelIsotonicScaling = labelIsotonicScaling;
    }

    @Override
    public IMLGradientBoosting getModel() {
        return imlGradientBoosting;
    }

    @Override
    public MultiLabel predict(Vector vector) {
        MultiLabel prediction = new MultiLabel();
        double[] uncalibrated = imlGradientBoosting.predictClassProbs(vector);
        double[] calibrated = labelIsotonicScaling.calibratedClassProbs(uncalibrated);
        for (int k=0;k<getNumClasses();k++){
            if (calibrated[k] > 0.5){
                prediction.addLabel(k);
            }
        }
        return prediction;
    }

}
