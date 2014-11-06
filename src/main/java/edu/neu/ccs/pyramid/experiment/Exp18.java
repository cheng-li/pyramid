package edu.neu.ccs.pyramid.experiment;

import edu.neu.ccs.pyramid.configuration.Config;
import edu.neu.ccs.pyramid.dataset.*;
import edu.neu.ccs.pyramid.elasticsearch.SingleLabelIndex;
import edu.neu.ccs.pyramid.elasticsearch.TermStat;
import edu.neu.ccs.pyramid.feature.*;
import org.elasticsearch.action.search.SearchResponse;
import org.elasticsearch.search.SearchHit;

import java.util.Arrays;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import java.util.stream.Collectors;

/**
 * single label dataset, start with initial features + all unigram features
 * use picked unigram features as seeds
 * Created by chengli on 11/6/14.
 */
public class Exp18 {


    private static boolean matchPrefixes(String name, Set<String> prefixes){
        for (String prefix: prefixes){
            if (name.startsWith(prefix)){
                return true;
            }
        }
        return false;
    }

    /**
     *
     * @param config
     * @param index
     * @param featureMappers to be updated
     * @param ids pull features from train ids
     * @throws Exception
     */
    static void addInitialFeatures(Config config, SingleLabelIndex index,
                                   FeatureMappers featureMappers,
                                   String[] ids) throws Exception{
        String featureFieldPrefix = config.getString("index.featureFieldPrefix");
        Set<String> prefixes = Arrays.stream(featureFieldPrefix.split(",")).map(String::trim).collect(Collectors.toSet());

        Set<String> allFields = index.listAllFields();
        List<String> featureFields = allFields.stream().
                filter(field -> matchPrefixes(field,prefixes)).
                collect(Collectors.toList());
        System.out.println("all possible initial features:"+featureFields);

        for (String field: featureFields){
            String featureType = index.getFieldType(field);
            if (featureType.equalsIgnoreCase("string")){
                CategoricalFeatureMapperBuilder builder = new CategoricalFeatureMapperBuilder();
                builder.setFeatureName(field);
                builder.setStart(featureMappers.nextAvailable());
                builder.setSource("field");
                for (String id: ids){
                    String category = index.getStringField(id, field);
                    builder.addCategory(category);
                }
                boolean toAdd = true;
                CategoricalFeatureMapper mapper = builder.build();
                if (config.getBoolean("categFeature.filter")){
                    double threshold = config.getDouble("categFeature.percentThreshold");
                    int numCategories = mapper.getNumCategories();
                    if (numCategories> ids.length*threshold){
                        toAdd=false;
                        System.out.println("field "+field+" has too many categories "
                                +"("+numCategories+"), omitted.");
                    }
                }
                if(toAdd){
                    featureMappers.addMapper(mapper);
                }

            } else {
                NumericalFeatureMapperBuilder builder = new NumericalFeatureMapperBuilder();
                builder.setFeatureName(field);
                builder.setFeatureIndex(featureMappers.nextAvailable());
                builder.setSource("field");
                NumericalFeatureMapper mapper = builder.build();
                featureMappers.addMapper(mapper);
            }
        }
    }

    static List<String> gatherUnigrams(Config config, SingleLabelIndex index,
                                       String[] ids) throws Exception{
        int minDf = config.getInt("minDf");
        Set<String> unigrams = new HashSet<>();
        for (String id: ids) {
            Set<TermStat> termStats = index.getTermStats(id);
            termStats.stream().filter(termStat -> termStat.getDf() > minDf).forEach(termStat -> unigrams.add(termStat.getTerm()));
        }
        return unigrams.stream().sorted().collect(Collectors.toList());
    }

    static void addUnigramFeatures(FeatureMappers featureMappers,List<String> unigrams){
        for (String unigram: unigrams){
            int featureIndex = featureMappers.nextAvailable();
            NumericalFeatureMapper mapper = NumericalFeatureMapper.getBuilder().
                    setFeatureIndex(featureIndex).setFeatureName(unigram).
                    setSource("unigram_matching_score").build();
            featureMappers.addMapper(mapper);
        }
    }

    /**
     *
     * @param config
     * @param index
     * @param featureMappers
     * @param idTranslator separate for training and test
     * @param totalDim
     * @return
     * @throws Exception
     */
    static ClfDataSet loadData(Config config, SingleLabelIndex index,
                               FeatureMappers featureMappers,
                               IdTranslator idTranslator, int totalDim,
                               LabelTranslator labelTranslator) throws Exception{
        int numDataPoints = idTranslator.numData();
        int numClasses = config.getInt("numClasses");
        ClfDataSet dataSet = ClfDataSetBuilder.getBuilder()
                .numDataPoints(numDataPoints).numFeatures(totalDim)
                .numClasses(numClasses).dense(!config.getBoolean("featureMatrix.sparse"))
                .missingValue(config.getBoolean("featureMatrix.missingValue"))
                .build();

        for(int i=0;i<numDataPoints;i++){
            String dataIndexId = idTranslator.toExtId(i);
            int label = index.getLabel(dataIndexId);
            dataSet.setLabel(i,label);
        }

        String[] dataIndexIds = idTranslator.getAllExtIds();

        featureMappers.getCategoricalFeatureMappers().stream().parallel().
                forEach(categoricalFeatureMapper -> {
                    String featureName = categoricalFeatureMapper.getFeatureName();
                    String source = categoricalFeatureMapper.getSource();
                    if (source.equalsIgnoreCase("field")){
                        for (String id: dataIndexIds){
                            int algorithmId = idTranslator.toIntId(id);
                            String category = index.getStringField(id,featureName);
                            // might be a new category unseen in training
                            if (categoricalFeatureMapper.hasCategory(category)){
                                int featureIndex = categoricalFeatureMapper.getFeatureIndex(category);
                                dataSet.setFeatureValue(algorithmId,featureIndex,1);
                            }
                        }
                    }
                });


        featureMappers.getNumericalFeatureMappers().stream().parallel().
                forEach(numericalFeatureMapper -> {
                    String featureName = numericalFeatureMapper.getFeatureName();
                    String source = numericalFeatureMapper.getSource();
                    int featureIndex = numericalFeatureMapper.getFeatureIndex();

                    if (source.equalsIgnoreCase("field")){
                        for (String id: dataIndexIds){
                            int algorithmId = idTranslator.toIntId(id);
                            float value = index.getFloatField(id,featureName);
                            dataSet.setFeatureValue(algorithmId,featureIndex,value);
                        }
                    }

                    if (source.equalsIgnoreCase("unigram_matching_score")||source.equalsIgnoreCase("ngram_matching_score")){
                        SearchResponse response = null;
                        response = index.matchPhrase(index.getBodyField(), featureName, dataIndexIds, 0);

                        SearchHit[] hits = response.getHits().getHits();
                        for (SearchHit hit: hits){
                            String indexId = hit.getId();
                            float score = hit.getScore();
                            int algorithmId = idTranslator.toIntId(indexId);
                            dataSet.setFeatureValue(algorithmId,featureIndex,score);
                        }
                    }
                });
        DataSetUtil.setIdTranslator(dataSet, idTranslator);
        DataSetUtil.setLabelTranslator(dataSet, labelTranslator);
        return dataSet;
    }

}
