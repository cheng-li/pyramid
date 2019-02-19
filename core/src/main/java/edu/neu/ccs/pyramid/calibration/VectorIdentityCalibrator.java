package edu.neu.ccs.pyramid.calibration;


import org.apache.mahout.math.Vector;

import java.io.Serializable;


public class VectorIdentityCalibrator implements Serializable, VectorCalibrator {
    private static final long serialVersionUID = 1L;
    private int scoreIndex;

    public VectorIdentityCalibrator(int scoreIndex) {
        this.scoreIndex = scoreIndex;
    }

    @Override
    public double calibrate(Vector vector) {
        return vector.get(scoreIndex);
    }
}
