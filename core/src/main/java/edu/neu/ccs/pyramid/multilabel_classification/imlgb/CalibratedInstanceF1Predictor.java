package edu.neu.ccs.pyramid.multilabel_classification.imlgb;

import edu.neu.ccs.pyramid.calibration.StreamGenerator;
import edu.neu.ccs.pyramid.dataset.MultiLabel;
import edu.neu.ccs.pyramid.multilabel_classification.PluginPredictor;
import edu.neu.ccs.pyramid.multilabel_classification.plugin_rule.GeneralF1Predictor;
import edu.neu.ccs.pyramid.regression.Regressor;
import org.apache.mahout.math.Vector;

import java.util.List;

public class CalibratedInstanceF1Predictor implements PluginPredictor<IMLGradientBoosting> {
    private static final long serialVersionUID = 1L;
    private IMLGradientBoosting imlGradientBoosting;
    private StreamGenerator streamGenerator;
    private Regressor setCalibrator;

    public CalibratedInstanceF1Predictor(IMLGradientBoosting imlGradientBoosting, StreamGenerator streamGenerator, Regressor setCalibrator) {
        this.imlGradientBoosting = imlGradientBoosting;
        this.streamGenerator = streamGenerator;
        this.setCalibrator = setCalibrator;
    }

    @Override
    public IMLGradientBoosting getModel() {
        return imlGradientBoosting;
    }

    @Override
    public MultiLabel predict(Vector vector) {
        List<Double> supportProbs = streamGenerator.calibratedProbsForSupport(setCalibrator,vector);
        GeneralF1Predictor generalF1Predictor = new GeneralF1Predictor();
        return generalF1Predictor.predict(imlGradientBoosting.getNumClasses(),imlGradientBoosting.getAssignments(),supportProbs);
    }
}
