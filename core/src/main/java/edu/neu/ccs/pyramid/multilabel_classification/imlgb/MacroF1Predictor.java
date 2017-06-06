package edu.neu.ccs.pyramid.multilabel_classification.imlgb;

import edu.neu.ccs.pyramid.dataset.MultiLabel;
import edu.neu.ccs.pyramid.multilabel_classification.PluginPredictor;
import edu.neu.ccs.pyramid.multilabel_classification.thresholding.TunedMarginalClassifier;
import org.apache.mahout.math.Vector;

/**
 * Created by chengli on 5/29/16.
 */
public class MacroF1Predictor implements PluginPredictor<IMLGradientBoosting> {
    private static final long serialVersionUID = 1L;
    private IMLGradientBoosting imlGradientBoosting;
    private TunedMarginalClassifier tunedMarginalClassifier;

    public MacroF1Predictor(IMLGradientBoosting imlGradientBoosting, TunedMarginalClassifier tunedMarginalClassifier) {
        this.imlGradientBoosting = imlGradientBoosting;
        this.tunedMarginalClassifier = tunedMarginalClassifier;
    }

    @Override
    public IMLGradientBoosting getModel() {
        return imlGradientBoosting;
    }

    @Override
    public MultiLabel predict(Vector vector) {
        return tunedMarginalClassifier.predict(vector);
    }
}
