package edu.neu.ccs.pyramid.regression.m_boost;

import edu.neu.ccs.pyramid.optimization.gradient_boosting.GradientBoosting;
import edu.neu.ccs.pyramid.regression.Regressor;
import org.apache.mahout.math.Vector;

/**
 * Created by chengli on 10/8/16.
 */
public class MBoost extends GradientBoosting implements Regressor {
    private static final long serialVersionUID = 1L;

    public MBoost() {
        super(1);
    }

    @Override
    public double predict(Vector vector) {
        return getEnsemble(0).score(vector);
    }
}
