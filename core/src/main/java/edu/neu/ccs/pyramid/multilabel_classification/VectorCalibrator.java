package edu.neu.ccs.pyramid.multilabel_classification;

import org.apache.mahout.math.Vector;

public interface VectorCalibrator {
    double calibrate(Vector vector);
}
