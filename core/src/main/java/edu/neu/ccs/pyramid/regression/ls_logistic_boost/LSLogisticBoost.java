package edu.neu.ccs.pyramid.regression.ls_logistic_boost;

import edu.neu.ccs.pyramid.optimization.gradient_boosting.GradientBoosting;
import edu.neu.ccs.pyramid.regression.Regressor;
import edu.neu.ccs.pyramid.util.MathUtil;
import edu.neu.ccs.pyramid.util.Sigmoid;
import org.apache.mahout.math.Vector;

/**
 * logistic boost as a regressor optimized by squared loss
 */
public class LSLogisticBoost extends GradientBoosting implements Regressor {
    private static final long serialVersionUID = 1L;

    public LSLogisticBoost() {
        super(1);
    }

    @Override
    public double predict(Vector vector) {
        return Sigmoid.sigmoid(getEnsemble(0).score(vector));
    }

}
