package edu.neu.ccs.pyramid.multilabel_classification.imlgb;

import edu.neu.ccs.pyramid.dataset.MultiLabel;
import edu.neu.ccs.pyramid.multilabel_classification.AbstractPlugIn;
import edu.neu.ccs.pyramid.multilabel_classification.plugin_rule.F1Predictor;
import org.apache.mahout.math.Vector;

import java.util.Arrays;
import java.util.List;
import java.util.stream.Collectors;

/**
 * Created by chengli on 4/19/16.
 */
public class PlugInF1 extends AbstractPlugIn {
    private static final long serialVersionUID = 1L;
    private IMLGradientBoosting imlGradientBoosting;

    public PlugInF1(IMLGradientBoosting imlGradientBoosting) {
        super(imlGradientBoosting);
        this.imlGradientBoosting = imlGradientBoosting;
    }

    @Override
    public MultiLabel predict(Vector vector) {
        double[] probs = imlGradientBoosting.predictAllAssignmentProbsWithConstraint(vector);
        List<Double> probList = Arrays.stream(probs).mapToObj(a->a).collect(Collectors.toList());
        return F1Predictor.predict(imlGradientBoosting.getNumClasses(),imlGradientBoosting.getAssignments(),probList);
    }
}
