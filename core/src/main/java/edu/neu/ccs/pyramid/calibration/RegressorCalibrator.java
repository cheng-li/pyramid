package edu.neu.ccs.pyramid.calibration;

import edu.neu.ccs.pyramid.regression.Regressor;
import org.apache.mahout.math.Vector;

public class RegressorCalibrator implements VectorCalibrator{

    Regressor regressor;

    public RegressorCalibrator(Regressor regressor) {
        this.regressor = regressor;
    }

    @Override
    public double calibrate(Vector vector) {
        double score = regressor.predict(vector);
        if (score>1){
            score=1;
        }

        if (score<0){
            score=0;
        }
        return score;
    }
}
