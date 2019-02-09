package edu.neu.ccs.pyramid.calibration;

import edu.neu.ccs.pyramid.dataset.MultiLabel;
import edu.neu.ccs.pyramid.feature.Feature;
import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.Vector;

import java.util.ArrayList;
import java.util.List;

public class CardFeatureExtractor implements PredictionFeatureExtractor{
    private static final long serialVersionUID = 1L;

    @Override
    public Vector extractFeatures(PredictionCandidate prediction) {
        Vector vector = new DenseVector(1);
        vector.set(0,prediction.multiLabel.getNumMatchedLabels());
        return vector;
    }

    @Override
    public int[] featureMonotonicity() {
        int[] mono = {0};
        return mono;
    }

    @Override
    public List<Feature> getNames() {
        Feature feature = new Feature();
        feature.setName("card");
        List<Feature> features = new ArrayList<>();
        features.add(feature);
        return features;
    }
}
