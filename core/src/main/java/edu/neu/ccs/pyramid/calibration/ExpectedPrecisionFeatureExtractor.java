package edu.neu.ccs.pyramid.calibration;

import edu.neu.ccs.pyramid.dataset.MultiLabel;
import edu.neu.ccs.pyramid.eval.FMeasure;
import edu.neu.ccs.pyramid.eval.Precision;
import edu.neu.ccs.pyramid.feature.Feature;
import edu.neu.ccs.pyramid.util.Pair;
import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.Vector;

import java.util.ArrayList;
import java.util.List;

public class ExpectedPrecisionFeatureExtractor implements PredictionFeatureExtractor {
    private static final long serialVersionUID = 1L;

    @Override
    public Vector extractFeatures(PredictionCandidate predictionCandidate) {
        MultiLabel prediction = predictionCandidate.multiLabel;
        double expectation = 0;
        List<Pair<MultiLabel,Double>> sparseJoint = predictionCandidate.sparseJoint;
        for (Pair<MultiLabel,Double> pair: sparseJoint){
            MultiLabel multiLabel = pair.getFirst();
            double prob = pair.getSecond();
            expectation += Precision.precision(multiLabel,prediction)*prob;
        }
        Vector vector = new DenseVector(1);
        vector.set(0,expectation);
        return vector;
    }

    @Override
    public int[] featureMonotonicity() {
        int[] mono = {1};
        return mono;
    }

    @Override
    public List<Feature> getNames() {
        Feature feature = new Feature();
        feature.setName("expected Precision");
        List<Feature> features = new ArrayList<>();
        features.add(feature);
        return features;
    }
}
