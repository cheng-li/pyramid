package edu.neu.ccs.pyramid.calibration;

import edu.neu.ccs.pyramid.dataset.MultiLabel;
import edu.neu.ccs.pyramid.feature.Feature;
import edu.neu.ccs.pyramid.util.ArrayUtil;
import edu.neu.ccs.pyramid.util.Vectors;
import org.apache.mahout.math.Vector;

import java.util.ArrayList;
import java.util.List;

public class CombinedPredictionFeatureExtractor implements PredictionFeatureExtractor{
    private static final long serialVersionUID = 1L;

    List<PredictionFeatureExtractor> list;

    public CombinedPredictionFeatureExtractor(List<PredictionFeatureExtractor> list) {
        this.list = list;
    }

    @Override
    public Vector extractFeatures(PredictionCandidate prediction) {
        List<Vector> vectors = new ArrayList<>();
        for (PredictionFeatureExtractor predictionFeatureExtractor: list){
            vectors.add(predictionFeatureExtractor.extractFeatures(prediction));
        }
        return Vectors.conatenateToSparseRandom(vectors);
    }

    @Override
    public int[] featureMonotonicity() {
        List<int[]> monos = new ArrayList<>();
        for (PredictionFeatureExtractor extractor: list){
            monos.add(extractor.featureMonotonicity());
        }
        return ArrayUtil.concatenate(monos);
    }

    @Override
    public List<Feature> getNames() {
        List<Feature> features = new ArrayList<>();
        for (PredictionFeatureExtractor extractor: list){
            features.addAll(extractor.getNames());
        }
        return features;
    }
}
