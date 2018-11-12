package edu.neu.ccs.pyramid.calibration;

import java.io.Serializable;
import java.util.stream.IntStream;

public interface LabelCalibrator extends Serializable {
    public double calibratedClassProb(double prob, int labelIndex);

    public default double[] calibratedClassProbs(double[]probs){
        return IntStream.range(0, probs.length).mapToDouble(j->calibratedClassProb(probs[j], j)).toArray();
    }
}
