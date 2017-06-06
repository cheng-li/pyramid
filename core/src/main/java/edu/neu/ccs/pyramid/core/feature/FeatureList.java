package edu.neu.ccs.pyramid.core.feature;

import java.io.*;
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

    public Feature get(int featureIndex){
        return features.get(featureIndex);
    }

    public int size(){
        return features.size();
    }

    public int nextAvailable(){
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

    public FeatureList deepCopy() throws IOException, ClassNotFoundException {
        ByteArrayOutputStream baos = new ByteArrayOutputStream();
        ObjectOutputStream oos = new ObjectOutputStream(baos);
        oos.writeObject(this);

        ByteArrayInputStream bais = new ByteArrayInputStream(baos.toByteArray());
        ObjectInputStream ois = new ObjectInputStream(bais);
        return (FeatureList) ois.readObject();
    }

    public void clearInices(){
        for (Feature feature: features){
            feature.clearIndex();
        }
    }

}
