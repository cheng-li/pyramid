package edu.neu.ccs.pyramid.ranking;

import edu.neu.ccs.pyramid.optimization.gradient_boosting.GradientBoosting;
import edu.neu.ccs.pyramid.regression.Regressor;
import org.apache.mahout.math.Vector;

public class LambdaMART extends GradientBoosting implements Regressor {

    public LambdaMART() {
        super(1);
    }

    @Override
    public double predict(Vector vector) {
        return score(vector,0);
    }
}
