package edu.neu.ccs.pyramid.calibration;

import edu.neu.ccs.pyramid.dataset.MultiLabel;
import edu.neu.ccs.pyramid.eval.FMeasure;
import edu.neu.ccs.pyramid.feature.Feature;
import edu.neu.ccs.pyramid.multilabel_classification.DynamicProgramming;
import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.Vector;

import java.util.ArrayList;
import java.util.List;

public class ExpectedF1FeatureExtractor implements PredictionFeatureExtractor {
    private static final long serialVersionUID = 1L;

    @Override
    public Vector extractFeatures(PredictionCandidate predictionCandidate) {
        MultiLabel prediction = predictionCandidate.multiLabel;
        double[] calibratedLabelProbs = predictionCandidate.labelProbs;
        double expectation = 0;
        DynamicProgramming dynamicProgramming = new DynamicProgramming(calibratedLabelProbs);
        for (int i=0;i<50;i++){
            DynamicProgramming.Candidate candidate = dynamicProgramming.nextHighest();
            MultiLabel multiLabel = candidate.getMultiLabel();
            double prob = candidate.getProbability();
            expectation += FMeasure.f1(prediction,multiLabel)*prob;
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
        feature.setName("expected F1");
        List<Feature> features = new ArrayList<>();
        features.add(feature);
        return features;
    }
}
