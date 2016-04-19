package edu.neu.ccs.pyramid.multilabel_classification.imlgb;

import edu.neu.ccs.pyramid.dataset.MultiLabel;
import edu.neu.ccs.pyramid.multilabel_classification.AbstractPlugIn;
import org.apache.mahout.math.Vector;

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
        return null;
    }
}
