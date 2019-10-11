package edu.neu.ccs.pyramid.calibration;

import org.apache.mahout.math.Vector;

import java.io.Serializable;

public class ZeroCalibrator implements Serializable, VectorCalibrator {

    private static final long serialVersionUID = 1L;

    @Override
    public double calibrate(Vector vector) {
        return 0;
    }
}
