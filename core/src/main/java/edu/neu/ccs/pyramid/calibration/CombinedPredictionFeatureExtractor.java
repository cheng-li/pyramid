package edu.neu.ccs.pyramid.calibration;

import edu.neu.ccs.pyramid.dataset.MultiLabel;
import edu.neu.ccs.pyramid.feature.Feature;
import edu.neu.ccs.pyramid.util.Vectors;
import org.apache.mahout.math.Vector;

import java.util.ArrayList;
import java.util.List;

public class CombinedPredictionFeatureExtractor implements PredictionFeatureExtractor{
    List<PredictionFeatureExtractor> list;


    @Override
    public Vector extractFeatures(MultiLabel prediction) {
        List<Vector> vectors = new ArrayList<>();
        for (PredictionFeatureExtractor predictionFeatureExtractor: list){
            vectors.add(predictionFeatureExtractor.extractFeatures(prediction));
        }
        return Vectors.conatenateToSparseRandom(vectors);
    }

    @Override
    public int[] featureMonotonicity() {
        return new int[0];
    }

    @Override
    public List<Feature> getNames() {
        return null;
    }
}
