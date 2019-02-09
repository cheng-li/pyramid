package edu.neu.ccs.pyramid.calibration;

import edu.neu.ccs.pyramid.dataset.MultiLabel;
import edu.neu.ccs.pyramid.feature.Feature;
import edu.neu.ccs.pyramid.multilabel_classification.MultiLabelClassifier;
import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.Vector;

import java.util.ArrayList;
import java.util.List;

public class BRProbFeatureExtractor implements PredictionFeatureExtractor{
    private static final long serialVersionUID = 1L;

    @Override
    public Vector extractFeatures(PredictionCandidate predictionCandidate) {
        MultiLabel prediction = predictionCandidate.multiLabel;
        double[] calibratedLabelProbs = predictionCandidate.labelProbs;
        double prod = 1;
        for (int l=0;l<calibratedLabelProbs.length;l++){
            if (prediction.matchClass(l)){
                prod *= calibratedLabelProbs[l];
            } else {
                prod *= 1-calibratedLabelProbs[l];
            }
        }
        Vector vector = new DenseVector(1);
        vector.set(0,prod);
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
        feature.setName("BR prob");
        List<Feature> features = new ArrayList<>();
        features.add(feature);
        return features;
    }
}
