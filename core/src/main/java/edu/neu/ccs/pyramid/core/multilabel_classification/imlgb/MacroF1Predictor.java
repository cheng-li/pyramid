package edu.neu.ccs.pyramid.core.multilabel_classification.imlgb;

import edu.neu.ccs.pyramid.core.multilabel_classification.thresholding.TunedMarginalClassifier;
import edu.neu.ccs.pyramid.core.dataset.MultiLabel;
import edu.neu.ccs.pyramid.core.multilabel_classification.PluginPredictor;
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
