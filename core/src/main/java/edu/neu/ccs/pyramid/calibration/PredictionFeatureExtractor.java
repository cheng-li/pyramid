package edu.neu.ccs.pyramid.calibration;

import edu.neu.ccs.pyramid.dataset.MultiLabel;
import edu.neu.ccs.pyramid.feature.Feature;
import org.apache.mahout.math.Vector;

import java.util.List;

public interface PredictionFeatureExtractor {
    Vector extractFeatures(MultiLabel prediction);
    int[] featureMonotonicity();
    List<Feature> getNames();


}
