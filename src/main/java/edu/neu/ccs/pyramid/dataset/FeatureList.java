package edu.neu.ccs.pyramid.dataset;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.List;

/**
 * Created by chengli on 3/6/15.
 */
public class FeatureList implements Serializable {
    private static final long serialVersionUID = 1L;
    private List<Feature> features;

    public FeatureList() {
        this.features = new ArrayList<>();
    }

    public FeatureList(List<Feature> features){
        this();
        addAll(features);
    }

    public int size(){
        return features.size();
    }

    // keep indices in order
    public synchronized void add(Feature feature){
        int index = features.size();
        feature.setIndex(index);
        features.add(feature);
    }

    public synchronized void addAll(List<Feature> features){
        features.forEach(this::add);
    }

    public List<Feature> getAll() {
        return features;
    }

}
