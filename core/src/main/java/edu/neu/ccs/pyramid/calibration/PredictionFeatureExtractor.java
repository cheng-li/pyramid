package edu.neu.ccs.pyramid.calibration;

import edu.neu.ccs.pyramid.dataset.MultiLabel;
import edu.neu.ccs.pyramid.feature.Feature;
import org.apache.mahout.math.Vector;

import java.io.Serializable;
import java.util.List;

public interface PredictionFeatureExtractor extends Serializable {
    Vector extractFeatures(PredictionCandidate predictionCandidate);
    int[] featureMonotonicity();
    List<Feature> getNames();


}
