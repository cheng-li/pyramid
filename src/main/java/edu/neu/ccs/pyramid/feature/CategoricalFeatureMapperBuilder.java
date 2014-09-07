package edu.neu.ccs.pyramid.feature;

import java.util.*;

/**
 * Created by chengli on 9/7/14.
 */
public class CategoricalFeatureMapperBuilder {
    private Set<String> categories;
    private int start;
    private boolean startSet = false;
    private String featureName = "no name";
    private String source = "unknown";

    public CategoricalFeatureMapperBuilder() {
        this.categories = new HashSet<>();
    }

    public CategoricalFeatureMapperBuilder addCategory(String category){
        this.categories.add(category);
        return this;
    }

    public CategoricalFeatureMapperBuilder setStart(int start) {
        this.start = start;
        this.startSet = true;
        return this;
    }

    public CategoricalFeatureMapperBuilder setFeatureName(String featureName) {
        this.featureName = featureName;
        return this;
    }

    public CategoricalFeatureMapperBuilder setSource(String source) {
        this.source = source;
        return this;
    }

    public CategoricalFeatureMapper build(){
        if (!this.startSet){
            throw new RuntimeException("start is not set yet!");
        }
        Map<String, Integer> categoryIndexMap = new HashMap<>();
        Map<Integer, String> indexCategoryMap = new HashMap<>();
        List<String> sortedCats = new ArrayList<>(this.categories.size());
        sortedCats.addAll(this.categories);
        /**
         * not necessary, but looks better
         */
        Collections.sort(sortedCats);
        int pos = this.start;
        for (String category: sortedCats){
            categoryIndexMap.put(category,pos);
            indexCategoryMap.put(pos,category);
            pos += 1;
        }

        return new CategoricalFeatureMapper(this.featureName,
                this.source,
                this.start,this.start+this.categories.size()-1,
                categoryIndexMap,indexCategoryMap);
    }


}
