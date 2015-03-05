package edu.neu.ccs.pyramid.feature;

import edu.neu.ccs.pyramid.dataset.CategoricalFeature;

import java.util.*;
import java.util.concurrent.ConcurrentHashMap;

/**
 * Created by chengli on 3/4/15.
 */
public class CategoricalFeatureExpander {
    private Set<String> categories;
    private int start;
    private boolean startSet = false;
    private String variableName;
    private Map<String,String> commonSettings;



    public CategoricalFeatureExpander() {
        this.categories = Collections.newSetFromMap(new ConcurrentHashMap<String, Boolean>());
        this.commonSettings = new HashMap<>();
    }

    public void setVariableName(String variableName) {
        this.variableName = variableName;
    }

    public void addCategory(String category){
        this.categories.add(category);
    }

    public void setStart(int start) {
        this.start = start;
        this.startSet = true;
    }

    public void putSetting(String key, String value){
        this.commonSettings.put(key,value);
    }

    public List<CategoricalFeature> expand(){
        if (!this.startSet){
            throw new RuntimeException("start is not set yet!");
        }
        List<CategoricalFeature> features = new ArrayList<>();

        Map<String, Integer> categoryIndexMap = new HashMap<>();
        List<String> sortedCats = new ArrayList<>(this.categories.size());
        sortedCats.addAll(this.categories);
        /**
         * not necessary, but looks better
         */
        Collections.sort(sortedCats);
        int pos = this.start;
        for (String category: sortedCats){
            CategoricalFeature feature = new CategoricalFeature();
            feature.setVariableName(variableName);
            feature.setCategory(category);
            feature.setCategoryIndexMap(categoryIndexMap);
            feature.setNumCategories(categories.size());
            feature.setIndex(pos);
            feature.setName(variableName + "[" + category + "]");
            for (Map.Entry<String,String> entry: this.commonSettings.entrySet()){
                feature.getSettings().put(entry.getKey(),entry.getValue());
            }
            categoryIndexMap.put(category, pos);
            features.add(feature);
            pos += 1;
        }

        return features;
    }
}
