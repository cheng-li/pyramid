package edu.neu.ccs.pyramid.elasticsearch;

import edu.neu.ccs.pyramid.dataset.*;
import edu.neu.ccs.pyramid.feature.*;
import org.apache.mahout.math.RandomAccessSparseVector;
import org.apache.mahout.math.Vector;
import org.elasticsearch.action.search.SearchResponse;
import org.elasticsearch.search.SearchHit;

import java.util.Arrays;
import java.util.Map;
import java.util.stream.IntStream;

/**
 * Created by chengli on 3/5/15.
 */
public class FeatureLoader {

    // assume categorical featureList are stored contiguously
    public static void loadFeatures(ESIndex index, DataSet dataSet, FeatureList features,
                                    IdTranslator idTranslator){
        boolean[] toHandle = new boolean[features.size()];
        Arrays.fill(toHandle,true);
        for (int i=0;i<features.size();i++){
            Feature feature = features.get(i);
            if (toHandle[i]){
                if (feature instanceof CategoricalFeature){
                    int numCategories = ((CategoricalFeature) feature).getNumCategories();
                    // all categories are already handled
                    // don't check other categories
                    for (int pos = i+1;pos<i+numCategories;pos++){
                        toHandle[pos] = false;
                    }
                }
            }
        }

        IntStream.range(0,features.size()).parallel().
                filter(i -> toHandle[i])
                .forEach(i-> {
                    if (i%10000==0){
                        System.out.println("feature "+i);
                    }
                    Feature feature = features.get(i);
                    if (feature instanceof CategoricalFeature){
                        loadCategoricalFeature(index,dataSet,(CategoricalFeature)feature,idTranslator);
                    } else if (feature instanceof Ngram){
                        loadNgramFeature(index, dataSet, (Ngram)feature, idTranslator);
                    } else if (feature instanceof SpanNotNgram){
                        loadSpanNotNgramFeature(index, dataSet, (SpanNotNgram)feature, idTranslator);
                    } else {
                        loadNumericalFeature(index,dataSet,feature,idTranslator);
                    }
                });
    }

    public static void loadCategoricalFeature(ESIndex index, DataSet dataSet, CategoricalFeature feature,
                                              IdTranslator idTranslator){
        String[] dataIndexIds = idTranslator.getAllExtIds();
        String variableName = feature.getVariableName();
        Map<String,Integer> categoryIndexMap = feature.getCategoryIndexMap();
        String source = feature.getSettings().get("source");
        if (source.equals("field")){
            Arrays.stream(dataIndexIds).forEach(id -> {
                int algorithmId = idTranslator.toIntId(id);
                String category = index.getStringField(id, variableName);
                // might be a new category unseen in training
                if (categoryIndexMap.containsKey(category)) {
                    int featureIndex = categoryIndexMap.get(category);
                    dataSet.setFeatureValue(algorithmId, featureIndex, 1);
                }
            });
        }
    }

    public static void loadNgramFeature(ESIndex index, DataSet dataSet, Ngram feature,
                                        IdTranslator idTranslator){
        String[] dataIndexIds = idTranslator.getAllExtIds();
        int featureIndex = feature.getIndex();
            SearchResponse response = index.spanNear(feature, dataIndexIds);
            SearchHit[] hits = response.getHits().getHits();
        for (SearchHit hit: hits){
            String indexId = hit.getId();
            float score = hit.getScore();
            int algorithmId = idTranslator.toIntId(indexId);
            dataSet.setFeatureValue(algorithmId,featureIndex,score);
        }
    }

    public static void loadSpanNotNgramFeature(ESIndex index, DataSet dataSet, SpanNotNgram feature,
                                        IdTranslator idTranslator){
        String[] dataIndexIds = idTranslator.getAllExtIds();

        int featureIndex = feature.getIndex();

        SearchResponse response = index.spanNot(feature, dataIndexIds);
        SearchHit[] hits = response.getHits().getHits();
        for (SearchHit hit: hits){
            String indexId = hit.getId();
            float score = hit.getScore();
            int algorithmId = idTranslator.toIntId(indexId);
            dataSet.setFeatureValue(algorithmId,featureIndex,score);
        }
    }

    public static Vector loadNgramFeature(ESIndex index, Ngram feature, IdTranslator idTranslator ){
        String[] dataIndexIds = idTranslator.getAllExtIds();
        SearchResponse response = index.spanNear(feature, dataIndexIds);
        SearchHit[] hits = response.getHits().getHits();
        Vector vector = new RandomAccessSparseVector(idTranslator.numData());
        for (SearchHit hit: hits){
            String indexId = hit.getId();
            float score = hit.getScore();
            int algorithmId = idTranslator.toIntId(indexId);
            vector.set(algorithmId,score);
        }
        return vector;
    }


    public static void loadNumericalFeature(ESIndex index, DataSet dataSet, Feature feature,
                                              IdTranslator idTranslator){
        String[] dataIndexIds = idTranslator.getAllExtIds();
        String variableName = feature.getName();
        int featureIndex = feature.getIndex();
        String source = feature.getSettings().get("source");
        if (source.equals("field")){
            Arrays.stream(dataIndexIds).forEach(id -> {
                int algorithmId = idTranslator.toIntId(id);
                double value = index.getFloatField(id,variableName);
                dataSet.setFeatureValue(algorithmId, featureIndex, value);
            });
        }
    }
}
