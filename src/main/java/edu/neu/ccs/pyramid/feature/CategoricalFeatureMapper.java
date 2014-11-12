package edu.neu.ccs.pyramid.feature;

import java.io.Serializable;
import java.util.Map;

/**
 * todo pull start from a featureMappers
 * span one categorical feature into several binary features
 * Created by chengli on 7/28/14.
 */
public class CategoricalFeatureMapper implements Serializable {
    private static final long serialVersionUID = 1L;
    private String featureName;
    private String source;
    /**
     * first feature index
     */
    private int start;
    /**
     * last feature index
     * feature categories are stored in contiguous positions
     */
    private int end;
    private Map<String, Integer> categoryIndexMap;
    private Map<Integer, String> indexCategoryMap;

    public int getStart() {
        return start;
    }

    /**
     * inclusive
     * @return
     */
    public int getEnd() {
        return end;
    }

    public String getFeatureName() {
        return featureName;
    }

    public int getNumCategories(){
        return end-start+1;
    }

    public boolean hasCategory(String category){
        return this.categoryIndexMap.containsKey(category);
    }

    public int getFeatureIndex(String category){
        return this.categoryIndexMap.get(category);
    }

    public String getCategory(int featureIndex){
        return this.indexCategoryMap.get(featureIndex);
    }

    public String getSource() {
        return source;
    }

    public static CategoricalFeatureMapperBuilder getBuilder(){
        return new CategoricalFeatureMapperBuilder();
    }

    Map<Integer, String> getIndexCategoryMap() {
        return indexCategoryMap;
    }

    @Override
    public String toString() {
        return "CategoricalFeatureMapper{" +
                "featureName='" + featureName + '\'' +
                ", source='" + source + '\'' +
                ", start=" + start +
                ", end=" + end +
                ", categoryIndexMap=" + categoryIndexMap +
                ", indexCategoryMap=" + indexCategoryMap +
                '}';
    }

    protected CategoricalFeatureMapper(String featureName, String source,
                                       int start, int end,
                                       Map<String, Integer> categoryIndexMap,
                                       Map<Integer, String> indexCategoryMap) {
        this.featureName = featureName;
        this.source = source;
        this.start = start;
        this.end = end;
        this.categoryIndexMap = categoryIndexMap;
        this.indexCategoryMap = indexCategoryMap;
    }
}
