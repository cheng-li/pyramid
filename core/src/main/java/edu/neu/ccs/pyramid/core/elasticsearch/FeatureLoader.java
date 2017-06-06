package edu.neu.ccs.pyramid.core.elasticsearch;

import edu.neu.ccs.pyramid.core.dataset.DataSet;
import edu.neu.ccs.pyramid.core.dataset.IdTranslator;
import edu.neu.ccs.pyramid.core.feature.*;
import org.elasticsearch.action.search.SearchResponse;
import org.elasticsearch.search.SearchHit;

import java.util.Arrays;
import java.util.List;
import java.util.stream.IntStream;

/**
 * Created by chengli on 3/5/15.
 */
public class FeatureLoader {


    public static void loadFeatures(ESIndex index, DataSet dataSet, FeatureList features,
                                    IdTranslator idTranslator, MatchScoreType matchScoreType, String docFilter){
//        ProgressBar progressBar = new ProgressBar(features.size());
        IntStream.range(0,features.size())
        		.parallel()
                .forEach(i-> {
                    Feature feature = features.get(i);
                    if (feature instanceof CategoricalFeature){
                        loadCategoricalFeature(index,dataSet,(CategoricalFeature)feature,idTranslator, docFilter);
                    } else if (feature instanceof Ngram){
                        loadNgramFeature(index, dataSet, (Ngram)feature, idTranslator, matchScoreType, docFilter);
                    } else if (feature instanceof CodeDescription) {
                        loadCodeDesFeature(index, dataSet, feature, idTranslator, docFilter);
                    } else {
                        loadNumericalFeature(index,dataSet,feature,idTranslator);
                    }

//                    progressBar.incrementAndPrint();
                }
                );
//        System.out.println();
    }

    public static void loadCategoricalFeature(ESIndex index, DataSet dataSet, CategoricalFeature feature,
                                              IdTranslator idTranslator, String docFilter){
        String[] dataIndexIds = idTranslator.getAllExtIds();
        String variableName = feature.getVariableName();
        int featureIndex = feature.getIndex();

        List<String> matchedIds = null;
        try {
            matchedIds = index.termFilter(variableName,feature.getCategory(),docFilter,idTranslator.numData());
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
                                        IdTranslator idTranslator, MatchScoreType matchScoreType, String docFilter){
        switch (matchScoreType){
            case ES_ORIGINAL:
                loadNgramFeatureOriginal(index, dataSet, feature, idTranslator, docFilter);
                break;
            case BINARY:
                loadNgramFeatureBinary(index, dataSet, feature, idTranslator, docFilter);
                break;
            case FREQUENCY:
                loadNgramFeatureFrequency(index, dataSet, feature, idTranslator, docFilter);
                break;
            case TFIFL:
                loadNgramFeatureTFIFL(index, dataSet, feature, idTranslator, docFilter);
        }
    }


    private static void loadNgramFeatureOriginal(ESIndex index, DataSet dataSet, Ngram feature,
                                                 IdTranslator idTranslator, String docFilter){
        int featureIndex = feature.getIndex();
        SearchResponse response = index.spanNear(feature, docFilter, idTranslator.numData());
        SearchHit[] hits = response.getHits().getHits();
        for (SearchHit hit: hits){
            String indexId = hit.getId();
            float score = hit.getScore();
            int algorithmId = idTranslator.toIntId(indexId);
            dataSet.setFeatureValue(algorithmId,featureIndex,score);
        }
    }

    private static void loadNgramFeatureFrequency(ESIndex index, DataSet dataSet, Ngram feature,
                                                  IdTranslator idTranslator, String docFilter){
        int featureIndex = feature.getIndex();
        SearchResponse response = index.spanNearFrequency(feature, docFilter, idTranslator.numData());
        SearchHit[] hits = response.getHits().getHits();
        for (SearchHit hit: hits){
            String indexId = hit.getId();
            float score = hit.getScore();
            int algorithmId = idTranslator.toIntId(indexId);
            dataSet.setFeatureValue(algorithmId,featureIndex,score);
        }
    }

    // term frequency inverse field length
    // field storing the length of the body field should be called body_field_length
    private static void loadNgramFeatureTFIFL(ESIndex index, DataSet dataSet, Ngram feature,
                                              IdTranslator idTranslator, String docFilter){
        int featureIndex = feature.getIndex();
        SearchResponse response = index.spanNearFrequency(feature, docFilter, idTranslator.numData());
        SearchHit[] hits = response.getHits().getHits();
        String field = feature.getField();
        String lengthField = field+"_"+"field_length";
        for (SearchHit hit: hits){
            String indexId = hit.getId();
            float score = hit.getScore();
            float docLength = index.getFloatField(indexId,lengthField);
            double s = score/docLength;
            int algorithmId = idTranslator.toIntId(indexId);
            dataSet.setFeatureValue(algorithmId,featureIndex,s);
        }
    }

    public static void loadNgramFeatureBinary(ESIndex index, DataSet dataSet, Ngram feature,
                                              IdTranslator idTranslator, String docFilter){
        int featureIndex = feature.getIndex();
        SearchResponse response = index.spanNear(feature, docFilter, idTranslator.numData());
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

//    public static void loadSpanNotNgramFeature(ESIndex index, DataSet dataSet, SpanNotNgram feature,
//                                        IdTranslator idTranslator){
//        String[] dataIndexIds = idTranslator.getAllExtIds();
//
//        int featureIndex = feature.getIndex();
//
//        SearchResponse response = index.spanNot(feature, dataIndexIds);
//        SearchHit[] hits = response.getHits().getHits();
//        for (SearchHit hit: hits){
//            String indexId = hit.getId();
//            float score = hit.getScore();
//            int algorithmId = idTranslator.toIntId(indexId);
//            dataSet.setFeatureValue(algorithmId,featureIndex,score);
//        }
//    }

//    public static Vector loadNgramFeature(ESIndex index, Ngram feature, IdTranslator idTranslator ){
//        String[] dataIndexIds = idTranslator.getAllExtIds();
//        SearchResponse response = index.spanNear(feature, dataIndexIds);
//        SearchHit[] hits = response.getHits().getHits();
//        Vector vector = new RandomAccessSparseVector(idTranslator.numData());
//        for (SearchHit hit: hits){
//            String indexId = hit.getId();
//            float score = hit.getScore();
//            int algorithmId = idTranslator.toIntId(indexId);
//            vector.set(algorithmId,score);
//        }
//        return vector;
//    }


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


    private static void loadCodeDesFeature(ESIndex index, DataSet dataSet, Feature feature,
                                           IdTranslator idTranslator, String docFilter){
        String[] dataIndexIds = idTranslator.getAllExtIds();
        int featureIndex = feature.getIndex();
        CodeDescription codeDescription = (CodeDescription)(feature);
        SearchResponse response = index.minimumShouldMatch(codeDescription.getDescription(), codeDescription.getField(), codeDescription.getPercentage(), idTranslator.numData(), docFilter);
        SearchHit[] hits = response.getHits().getHits();
        for (SearchHit hit: hits){
            String indexId = hit.getId();
            float score = hit.getScore();
            int algorithmId = idTranslator.toIntId(indexId);
            dataSet.setFeatureValue(algorithmId,featureIndex,score);
        }

    }


    public static enum MatchScoreType{
        ES_ORIGINAL, BINARY, FREQUENCY, TFIFL
    }
}
