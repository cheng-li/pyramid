package edu.neu.ccs.pyramid.calibration;

import org.apache.mahout.math.Vector;

import java.io.Serializable;

public interface VectorCalibrator extends Serializable {
    double calibrate(Vector vector);
}
