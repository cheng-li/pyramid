package edu.neu.ccs.pyramid.calibration;

import edu.neu.ccs.pyramid.dataset.MultiLabel;
import edu.neu.ccs.pyramid.dataset.MultiLabelClfDataSet;
import edu.neu.ccs.pyramid.feature.Feature;
import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.Vector;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class PriorFeatureExtractor implements PredictionFeatureExtractor {
    private Map<MultiLabel,Double> priors;

    public PriorFeatureExtractor(MultiLabelClfDataSet dataSet){
        priors = new HashMap<>();
        for (MultiLabel multiLabel : dataSet.getMultiLabels()) {
            double p = priors.getOrDefault(multiLabel, 0.0);
            priors.put(multiLabel, p + 1.0 / dataSet.getNumDataPoints());
        }
    }

    public PriorFeatureExtractor(Map<MultiLabel, Double> priors) {
        this.priors = priors;
    }

    @Override
    public List<Feature> getNames() {
        Feature feature = new Feature();
        feature.setName("setPrior");
        List<Feature> features = new ArrayList<>();
        features.add(feature);
        return features;
    }

    @Override
    public Vector extractFeatures(MultiLabel prediction) {
        Vector vector = new DenseVector(1);
        double prior = priors.getOrDefault(prediction,0.0);
        vector.set(0,prior);
        return vector;
    }

    @Override
    public int[] featureMonotonicity() {
        int[] mono = {1};
        return mono;
    }
}
