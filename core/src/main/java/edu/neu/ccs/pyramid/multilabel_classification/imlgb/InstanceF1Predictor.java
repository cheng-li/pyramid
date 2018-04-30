package edu.neu.ccs.pyramid.multilabel_classification.imlgb;

import edu.neu.ccs.pyramid.dataset.MultiLabel;
import edu.neu.ccs.pyramid.multilabel_classification.PluginPredictor;
import edu.neu.ccs.pyramid.multilabel_classification.plugin_rule.GeneralF1Predictor;
import org.apache.mahout.math.Vector;

import java.util.Arrays;
import java.util.List;
import java.util.stream.Collectors;

/**
 * Created by chengli on 4/19/16.
 */
public class InstanceF1Predictor implements PluginPredictor<IMLGradientBoosting> {
    private static final long serialVersionUID = 1L;
    private IMLGradientBoosting imlGradientBoosting;
    /**
     * exponentiate probabilities
     */
    private double exponent=1;

    public InstanceF1Predictor(IMLGradientBoosting imlGradientBoosting) {
        this.imlGradientBoosting = imlGradientBoosting;
    }


    public void setExponent(double exponent) {
        this.exponent = exponent;
    }

    @Override
    public IMLGradientBoosting getModel() {
        return imlGradientBoosting;
    }


    @Override
    public MultiLabel predict(Vector vector) {
        double[] probs = imlGradientBoosting.predictAllAssignmentProbsWithConstraint(vector);
        List<Double> probList = Arrays.stream(probs).mapToObj(a->Math.pow(a,exponent)).collect(Collectors.toList());
        GeneralF1Predictor generalF1Predictor = new GeneralF1Predictor();
        return generalF1Predictor.predict(imlGradientBoosting.getNumClasses(),imlGradientBoosting.getAssignments(),probList);
    }
}
