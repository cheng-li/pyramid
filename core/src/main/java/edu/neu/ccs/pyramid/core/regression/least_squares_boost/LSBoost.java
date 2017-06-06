package edu.neu.ccs.pyramid.core.regression.least_squares_boost;

import edu.neu.ccs.pyramid.core.optimization.gradient_boosting.GradientBoosting;
import edu.neu.ccs.pyramid.core.regression.Regressor;
import org.apache.mahout.math.Vector;


/**
 * Created by chengli on 6/3/15.
 */
public class LSBoost extends GradientBoosting implements Regressor {
    private static final long serialVersionUID = 1L;

    public LSBoost() {
        super(1);
    }

    @Override
    public double predict(Vector vector) {
        return getEnsemble(0).score(vector);
    }

}
