package edu.neu.ccs.pyramid.calibration;

import edu.neu.ccs.pyramid.dataset.MultiLabel;
import org.apache.mahout.math.Vector;

public class PredictionCandidate {
    public Vector x;
    public MultiLabel multiLabel;
    public double[] labelProbs;

}
