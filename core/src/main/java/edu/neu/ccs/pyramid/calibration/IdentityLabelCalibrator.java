package edu.neu.ccs.pyramid.calibration;

public class IdentityLabelCalibrator implements LabelCalibrator {
    @Override
    public double calibratedClassProb(double prob, int labelIndex) {
        return prob;
    }
}
