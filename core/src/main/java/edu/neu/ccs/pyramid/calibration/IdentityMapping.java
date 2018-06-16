package edu.neu.ccs.pyramid.calibration;

import edu.neu.ccs.pyramid.feature.FeatureList;
import edu.neu.ccs.pyramid.regression.Regressor;
import org.apache.mahout.math.Vector;

public class IdentityMapping implements Regressor {
    @Override
    public double predict(Vector vector) {
        return vector.get(0);
    }

    @Override
    public FeatureList getFeatureList() {
        return null;
    }
}
