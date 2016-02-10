package edu.neu.ccs.pyramid.elasticsearch;

import edu.neu.ccs.pyramid.dataset.*;
import edu.neu.ccs.pyramid.feature.*;
import edu.neu.ccs.pyramid.util.ProgressBar;
import org.apache.mahout.math.RandomAccessSparseVector;
import org.apache.mahout.math.Vector;
import org.elasticsearch.action.search.SearchResponse;
import org.elasticsearch.search.SearchHit;

import java.util.Arrays;
import java.util.List;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.stream.IntStream;

/**
 * Created by chengli on 3/5/15.
 */
public class FeatureLoader {

    public static void loadFeatures(ESIndex index, DataSet dataSet, FeatureList features,
                                    IdTranslator idTranslator){
        loadFeatures(index, dataSet, features, idTranslator, MatchScoreType.ES_ORIGINAL);

    }

    public static void loadFeatures(ESIndex index, DataSet dataSet, FeatureList features,
                                    IdTranslator idTranslator, MatchScoreType matchScoreType){
        ProgressBar progressBar = new ProgressBar(features.size());
        IntStream.range(0,features.size()).parallel()
                .forEach(i-> {
                    Feature feature = features.get(i);
                    if (feature instanceof CategoricalFeature){
                        loadCategoricalFeature(index,dataSet,(CategoricalFeature)feature,idTranslator);
                    } else if (feature instanceof Ngram){
                        loadNgramFeature(index, dataSet, (Ngram)feature, idTranslator, matchScoreType);
                    } else if (feature instanceof SpanNotNgram){
                        loadSpanNotNgramFeature(index, dataSet, (SpanNotNgram)feature, idTranslator);
                    } else {
                        loadNumericalFeature(index,dataSet,feature,idTranslator);
                    }

                    progressBar.incrementAndPrint();
                    System.out.println();
                });
    }

    public static void loadCategoricalFeature(ESIndex index, DataSet dataSet, CategoricalFeature feature,
                                              IdTranslator idTranslator){
        String[] dataIndexIds = idTranslator.getAllExtIds();
        String variableName = feature.getVariableName();
        int featureIndex = feature.getIndex();

        List<String> matchedIds = null;
        try {
            matchedIds = index.termFilter(variableName,feature.getCategory(),dataIndexIds);
        } catch (Exception e) {
            e.printStackTrace();
        }
        for (String matchedId: matchedIds){
            int algorithmId = idTranslator.toIntId(matchedId);
            dataSet.setFeatureValue(algorithmId,featureIndex,1);
        }

        List<String> docMissingField = index.docsWithFieldMissing(variableName,dataIndexIds);
        for (String extId: docMissingField){
            int algorithmId = idTranslator.toIntId(extId);
            dataSet.setFeatureValue(algorithmId,featureIndex,Double.NaN);
        }


    }

    public static void loadNgramFeature(ESIndex index, DataSet dataSet, Ngram feature,
                                        IdTranslator idTranslator, MatchScoreType matchScoreType){
        switch (matchScoreType){
            case ES_ORIGINAL:
                loadNgramFeatureOriginal(index, dataSet, feature, idTranslator);
                break;
            case BINARY:
                loadNgramFeatureBinary(index, dataSet, feature, idTranslator);
                break;
            case FREQUENCY:
                loadNgramFeatureFrequency(index, dataSet, feature, idTranslator);
        }
    }


    private static void loadNgramFeatureOriginal(ESIndex index, DataSet dataSet, Ngram feature,
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

    private static void loadNgramFeatureFrequency(ESIndex index, DataSet dataSet, Ngram feature,
                                                  IdTranslator idTranslator){
        String[] dataIndexIds = idTranslator.getAllExtIds();
        int featureIndex = feature.getIndex();
        SearchResponse response = index.spanNearFrequency(feature, dataIndexIds);
        SearchHit[] hits = response.getHits().getHits();
        for (SearchHit hit: hits){
            String indexId = hit.getId();
            float score = hit.getScore();
            int algorithmId = idTranslator.toIntId(indexId);
            dataSet.setFeatureValue(algorithmId,featureIndex,score);
        }
    }

    public static void loadNgramFeatureBinary(ESIndex index, DataSet dataSet, Ngram feature,
                                              IdTranslator idTranslator){
        String[] dataIndexIds = idTranslator.getAllExtIds();
        int featureIndex = feature.getIndex();
        SearchResponse response = index.spanNear(feature, dataIndexIds);
        SearchHit[] hits = response.getHits().getHits();
        for (SearchHit hit: hits){
            String indexId = hit.getId();
            float score = hit.getScore();
            int algorithmId = idTranslator.toIntId(indexId);
            if (score>0){
                score=1;
            }
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
                //may return NaN
                double value = index.getFloatField(id,variableName);
                dataSet.setFeatureValue(algorithmId, featureIndex, value);
            });
        }
    }

    public static enum MatchScoreType{
        ES_ORIGINAL, BINARY, FREQUENCY
    }
}
