package edu.neu.ccs.pyramid.experiment;


import edu.neu.ccs.pyramid.configuration.Config;
import edu.neu.ccs.pyramid.dataset.*;
import edu.neu.ccs.pyramid.elasticsearch.ESIndex;
import edu.neu.ccs.pyramid.elasticsearch.MultiLabelIndex;
import edu.neu.ccs.pyramid.elasticsearch.SingleLabelIndex;
import edu.neu.ccs.pyramid.eval.Accuracy;
import edu.neu.ccs.pyramid.feature.*;
import edu.neu.ccs.pyramid.feature_extraction.*;
import edu.neu.ccs.pyramid.multilabel_classification.imlgb.IMLGBConfig;
import edu.neu.ccs.pyramid.multilabel_classification.imlgb.IMLGradientBoosting;
import edu.neu.ccs.pyramid.util.Pair;
import edu.neu.ccs.pyramid.util.Sampling;
import org.apache.commons.lang3.time.StopWatch;
import org.elasticsearch.action.search.SearchResponse;
import org.elasticsearch.index.query.MatchQueryBuilder;
import org.elasticsearch.search.SearchHit;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.*;
import java.util.regex.Pattern;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

/**
 * feature extraction for multi label dataset
 * Created by chengli on 10/13/14.
 */
public class Exp15 {

    public static void main(String[] args) throws Exception{
        if (args.length !=1){
            throw new IllegalArgumentException("please specify the config file");
        }

        Config config = new Config(args[0]);
        System.out.println(config);

        MultiLabelIndex index = loadIndex(config);

        build(config,index);

        index.close();

    }

    static MultiLabelIndex loadIndex(Config config) throws Exception{
        MultiLabelIndex.Builder builder = new MultiLabelIndex.Builder()
                .setIndexName(config.getString("index.indexName"))
                .setClusterName(config.getString("index.clusterName"))
                .setClientType(config.getString("index.clientType"))
                .setExtMultiLabelField(config.getString("index.extMultiLabelField"))
                .setDocumentType(config.getString("index.documentType"));
        if (config.getString("index.clientType").equals("transport")){
            String[] hosts = config.getString("index.hosts").split(Pattern.quote(","));
            String[] ports = config.getString("index.ports").split(Pattern.quote(","));
            builder.addHostsAndPorts(hosts,ports);
        }
        MultiLabelIndex index = builder.build();
        System.out.println("index loaded");
        System.out.println("there are "+index.getNumDocs()+" documents in the index.");
//        for (int i=0;i<index.getNumDocs();i++){
//            System.out.println(i);
//            System.out.println(index.getLabel(""+i));
//        }
        return index;
    }

    static String[] sampleTrain(Config config, MultiLabelIndex index){
        int numDocsInIndex = index.getNumDocs();
        String[] trainIds = null;
        if (config.getString("split.fashion").equalsIgnoreCase("fixed")){
            String splitField = config.getString("index.splitField");
            trainIds = IntStream.range(0, numDocsInIndex).
                    filter(i -> index.getStringField("" + i, splitField).
                            equalsIgnoreCase("train")).
                    mapToObj(i -> "" + i).collect(Collectors.toList()).
                    toArray(new String[0]);
        } else if (config.getString("split.fashion").equalsIgnoreCase("random")){
            trainIds = Arrays.stream(Sampling.sampleByPercentage(numDocsInIndex, config.getDouble("split.random.trainPercentage"))).
                    mapToObj(i-> ""+i).
                    collect(Collectors.toList()).
                    toArray(new String[0]);
//            throw new IllegalArgumentException("random is not supported");
//            todo : how to do stratified sampling?
//            double trainPercentage = config.getDouble("split.random.trainPercentage");
//            int[] labels = new int[numDocsInIndex];
//            for (int i=0;i<labels.length;i++){
//                labels[i] = index.getLabel(""+i);
//            }
//            List<Integer> sample = Sampling.stratified(labels, trainPercentage);
//            trainIds = new String[sample.size()];
//            for (int i=0;i<trainIds.length;i++){
//                trainIds[i] = ""+sample.get(i);
//            }
        } else {
            throw new RuntimeException("illegal split fashion");
        }

        return trainIds;
    }

    static String[] sampleTest(int numDocsInIndex, String[] trainIndexIds){
        Set<String> test = new HashSet<>(numDocsInIndex);
        for (int i=0;i<numDocsInIndex;i++){
            test.add(""+i);
        }
        List<String> _trainIndexIds = new ArrayList<>(trainIndexIds.length);
        for (String id: trainIndexIds){
            _trainIndexIds.add(id);
        }

        test.removeAll(_trainIndexIds);
        return test.toArray(new String[0]);
    }

    static IdTranslator loadIdTranslator(String[] indexIds) throws Exception{
        IdTranslator idTranslator = new IdTranslator();
        for (int i=0;i<indexIds.length;i++){
            idTranslator.addData(i,""+indexIds[i]);
        }
        return idTranslator;
    }

    static void addInitialFeatures(Config config, MultiLabelIndex index,
                                   FeatureMappers featureMappers,
                                   String[] ids) throws Exception{
        String featureFieldPrefix = config.getString("index.featureFieldPrefix");

        Set<String> allFields = index.listAllFields();
        List<String> featureFields = allFields.stream().
                filter(field -> field.startsWith(featureFieldPrefix)).
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
                    // missing value is not a category
                    if (!category.equals(ESIndex.STRING_MISSING_VALUE)){
                        builder.addCategory(category);
                    }
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

    //todo keep track of feature types(numerical /binary)
    static MultiLabelClfDataSet loadData(Config config, MultiLabelIndex index,
                                         FeatureMappers featureMappers,
                                         IdTranslator idTranslator, int totalDim,
                                         LabelTranslator labelTranslator) throws Exception{
        int numDataPoints = idTranslator.numData();
        int numClasses = labelTranslator.getNumClasses();
        MultiLabelClfDataSet dataSet;
        if(config.getBoolean("featureMatrix.sparse")){
            dataSet= new SparseMLClfDataSet(numDataPoints,totalDim,numClasses);
        } else {
            dataSet= new DenseMLClfDataSet(numDataPoints,totalDim,numClasses);
        }
        for(int i=0;i<numDataPoints;i++){
            String dataIndexId = idTranslator.toExtId(i);
            List<String> extMultiLabel = index.getExtMultiLabel(dataIndexId);
            for (String extLabel: extMultiLabel){
                int intLabel = labelTranslator.toIntLabel(extLabel);
                dataSet.addLabel(i,intLabel);
            }
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
                            // if a value is missing, set nan
                            if (category.equals(ESIndex.STRING_MISSING_VALUE)){
                                for (int featureIndex=categoricalFeatureMapper.getStart();featureIndex<categoricalFeatureMapper.getEnd();featureIndex++){
                                    dataSet.setFeatureValue(algorithmId,featureIndex,Double.NaN);
                                }
                            }
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
                            // if it is missing, it is nan automatically
                            float value = index.getFloatField(id,featureName);
                            dataSet.setFeatureValue(algorithmId,featureIndex,value);
                        }
                    }

                    if (source.equalsIgnoreCase("matching_score")){
                        SearchResponse response = null;

                        //todo assume unigram, so slop doesn't matter
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

    static MultiLabelClfDataSet loadTrainData(Config config, MultiLabelIndex index, FeatureMappers featureMappers,
                                              IdTranslator idTranslator, LabelTranslator labelTranslator) throws Exception{
        int totalDim = config.getInt("maxNumColumns");
        System.out.println("creating training set");
        System.out.println("allocating "+totalDim+" columns for training set");
        MultiLabelClfDataSet dataSet = loadData(config,index,featureMappers,idTranslator,totalDim,labelTranslator);
        System.out.println("training set created");
        return dataSet;
    }

    static MultiLabelClfDataSet loadTestData(Config config, MultiLabelIndex index,
                                             FeatureMappers featureMappers, IdTranslator idTranslator,
                                             LabelTranslator labelTranslator) throws Exception{
        System.out.println("creating test set");

        int totalDim = featureMappers.getTotalDim();

        MultiLabelClfDataSet dataSet = loadData(config,index,featureMappers,idTranslator,totalDim,labelTranslator);
        System.out.println("test set created");
        return dataSet;
    }



//    static void showDistribution(Config config, ClfDataSet dataSet, Map<Integer, String> labelTranslator){
//        int numClasses = labelTranslator.size();
//        int[] counts = new int[numClasses];
//        int[] labels = dataSet.getLabels();
//        for (int i=0;i<dataSet.getNumDataPoints();i++){
//            int label = labels[i];
//            counts[label] += 1;
//
//        }
//        System.out.println("label distribution:");
//        for (int i=0;i<numClasses;i++){
//            System.out.print(i+"("+labelTranslator.get(i)+"):"+counts[i]+", ");
//        }
//        System.out.println("");
//    }

    static void saveDataSet(Config config, MultiLabelClfDataSet dataSet, String name) throws Exception{
        String archive = config.getString("archive.folder");
        File dataFile = new File(archive,name);
        TRECFormat.save(dataSet, dataFile);

        DataSetUtil.dumpDataSettings(dataSet,new File(dataFile,"data_settings.txt"));
        DataSetUtil.dumpFeatureSettings(dataSet,new File(dataFile,"feature_settings.txt"));
        System.out.println("data set saved to "+dataFile.getAbsolutePath());
    }

    static void dumpTrainFields(Config config, MultiLabelIndex index, IdTranslator idTranslator) throws Exception{
        String archive = config.getString("archive.folder");
        String trecFile = new File(archive,config.getString("archive.trainingSet")).getAbsolutePath();
        String file = new File(trecFile,"dumped_fields.txt").getAbsolutePath();
        dumpFields(config, index, idTranslator, file);
    }

    static void dumpTestFields(Config config, MultiLabelIndex index, IdTranslator idTranslator) throws Exception{
        String archive = config.getString("archive.folder");
        String trecFile = new File(archive,config.getString("archive.testSet")).getAbsolutePath();
        String file = new File(trecFile,"dumped_fields.txt").getAbsolutePath();
        dumpFields(config, index, idTranslator, file);
    }

    static void dumpFields(Config config, MultiLabelIndex index, IdTranslator idTranslator, String fileName) throws Exception{

        String[] fields = config.getString("archive.dumpedFields").split(",");
        int numDocs = idTranslator.numData();
        try(BufferedWriter bw = new BufferedWriter(new FileWriter(fileName))
        ){
            for (int intId=0;intId<numDocs;intId++){
                bw.write("intId=");
                bw.write(""+intId);
                bw.write(",");
                bw.write("extId=");
                String extId = idTranslator.toExtId(intId);
                bw.write(extId);
                bw.write(",");
                for (int i=0;i<fields.length;i++){
                    String field = fields[i];
                    bw.write(field+"=");
                    bw.write(index.getStringField(extId,field));
                    if (i!=fields.length-1){
                        bw.write(",");
                    }

                }
                bw.write("\n");
            }
        }

    }


    static void trainModel(Config config, MultiLabelClfDataSet dataSet, FeatureMappers featureMappers,
                           MultiLabelIndex index, IdTranslator trainIdTranslator) throws Exception{
        String archive = config.getString("archive.folder");
        int numIterations = config.getInt("train.numIterations");
        int numClasses = dataSet.getNumClasses();
        int numLeaves = config.getInt("train.numLeaves");
        double learningRate = config.getDouble("train.learningRate");
        int trainMinDataPerLeaf = config.getInt("train.minDataPerLeaf");


        String modelName = config.getString("archive.model");
        boolean overwriteModels = config.getBoolean("train.overwriteModels");
        int numDocsToSelect = config.getInt("extraction.numDocsToSelect");
        int numNgramsToExtract = config.getInt("extraction.numNgramsToExtract");
        double extractionFrequency = config.getDouble("extraction.frequency");
        if (extractionFrequency>1 || extractionFrequency<0){
            throw new IllegalArgumentException("0<=extraction.frequency<=1");
        }

        LabelTranslator labelTranslator = dataSet.getSetting().getLabelTranslator();

        StopWatch stopWatch = new StopWatch();
        stopWatch.start();

        System.out.println("extracting features");

        IMLGBConfig imlgbConfig = new IMLGBConfig.Builder(dataSet)
                .learningRate(learningRate).minDataPerLeaf(trainMinDataPerLeaf)
                .numLeaves(numLeaves)
                .build();

        IMLGradientBoosting boosting = new IMLGradientBoosting(numClasses);
        boosting.setPriorProbs(dataSet);
        boosting.setTrainConfig(imlgbConfig);


        TermTfidfSplitExtractor tfidfSplitExtractor = new TermTfidfSplitExtractor(index,
                trainIdTranslator,numNgramsToExtract).
                setMinDf(config.getInt("extraction.tfidfSplitExtractor.minDf")).
                setNumSurvivors(config.getInt("extraction.tfidfSplitExtractor.numSurvivors")).
                setMinDataPerLeaf(config.getInt("extraction.tfidfSplitExtractor.minDataPerLeaf"));

        PhraseSplitExtractor phraseSplitExtractor = new PhraseSplitExtractor(index,trainIdTranslator)
                .setMinDataPerLeaf(config.getInt("extraction.phraseSplitExtractor.minDataPerLeaf"))
                .setMinDf(config.getInt("extraction.phraseSplitExtractor.minDf"))
                .setTopN(config.getInt("extraction.phraseSplitExtractor.topN"));

        System.out.println("loading initial seeds...");
        DFStats dfStats = loadDFStats(index,trainIdTranslator,labelTranslator);
        List<Set<String>> seedsForAllClasses = new ArrayList<>();
        for (int i=0;i<numClasses;i++){
            Set<String> set = new HashSet<>();
            set.addAll(dfStats.getSortedTerms(i,config.getInt("extraction.seeds.minDf"),
                    config.getInt("extraction.seeds.numPerClass")));
            seedsForAllClasses.add(set);
        }

        System.out.println("seeds loaded");
        Set<String> blackList = new HashSet<>();

        //start the matrix with the seeds
        //may have duplicates, but should not be a big deal
        for(Set<String> seeds: seedsForAllClasses){
            for (String term: seeds){
                int featureIndex = featureMappers.nextAvailable();
                SearchResponse response = index.match(index.getBodyField(),
                        term,trainIdTranslator.getAllExtIds(), MatchQueryBuilder.Operator.AND);
                for (SearchHit hit: response.getHits().getHits()){
                    String indexId = hit.getId();
                    int algorithmId = trainIdTranslator.toIntId(indexId);
                    float score = hit.getScore();
                    dataSet.setFeatureValue(algorithmId, featureIndex,score);
                }

                NumericalFeatureMapper mapper = NumericalFeatureMapper.getBuilder().
                        setFeatureIndex(featureIndex).setFeatureName(term).
                        setSource("matching_score").build();
                featureMappers.addMapper(mapper);
                blackList.add(term);
            }
        }



//        //todo
//        List<Integer> validationSet = new ArrayList<>();
//        for (int i=0;i<trainIndex.getNumDocs();i++){
//            validationSet.add(i);
//        }

        for (int iteration=0;iteration<numIterations;iteration++){
            System.out.println("iteration "+iteration);
            boosting.calGradients();

            boolean condition1 = (featureMappers.getTotalDim()
                    +numNgramsToExtract*numClasses*2
                    +config.getInt("extraction.phraseSplitExtractor.topN")*numClasses*2
                    <dataSet.getNumFeatures());
            boolean condition2 = (Math.random()<extractionFrequency);
            //should start with some feature
            boolean condition3 = (iteration==0);


            boolean shouldExtractFeatures = condition1&&condition2||condition3;

            if (!shouldExtractFeatures){
                if (!condition1){
                    System.out.println("we have reached the max number of columns " +
                            "and will not extract new features");
                }

                if (!condition2){
                    System.out.println("no feature extraction is scheduled for this round");
                }
            }



            /**
             * from easy set
             */
            if (shouldExtractFeatures&&config.getBoolean("extraction.fromEasySet")){
                //generate easy set
                FocusSet focusSet = new FocusSet(numClasses);
                for (int k=0;k<numClasses;k++){
                    double[] gradient = boosting.getGradients(k);
                    Comparator<Pair<Integer,Double>> comparator = Comparator.comparing(Pair::getSecond);
                    List<Integer> easyExamples = IntStream.range(0,gradient.length)
                            .mapToObj(i -> new Pair<>(i,gradient[i]))
                            .filter(pair -> pair.getSecond()>0)
                            .sorted(comparator)
                            .limit(numDocsToSelect)
                            .map(Pair::getFirst)
                            .collect(Collectors.toList());
                    for(Integer doc: easyExamples){
                        focusSet.add(doc,k);
                    }
                }

                List<Integer> validationSet = focusSet.getAll();

                for (int k=0;k<numClasses;k++){
                    double[] allGradients = boosting.getGradients(k);
                    List<Double> gradientsForValidation = validationSet.stream()
                            .map(i -> allGradients[i]).collect(Collectors.toList());

                    List<String> goodTerms = null;

                    goodTerms = tfidfSplitExtractor.getGoodTerms(focusSet,
                            validationSet,
                            blackList, k, gradientsForValidation);


                    seedsForAllClasses.get(k).addAll(goodTerms);

                    List<String> focusSetIndexIds = focusSet.getDataClassK(k)
                            .parallelStream().map(trainIdTranslator::toExtId)
                            .collect(Collectors.toList());
                    System.out.println("easy set for class " +k+ "("+labelTranslator.toExtLabel(k)+ "):");
                    System.out.println(focusSetIndexIds.toString());
                    System.out.println("terms extracted from easy set for class " + k+" ("+labelTranslator.toExtLabel(k)+"):");
                    System.out.println(goodTerms);




                    //phrases
                    System.out.println("seeds for class " +k+ "("+labelTranslator.toExtLabel(k)+ "):");
                    System.out.println(seedsForAllClasses.get(k));
                    List<String> goodPhrases = phraseSplitExtractor.getGoodPhrases(focusSet,validationSet,blackList,k,
                            gradientsForValidation,seedsForAllClasses.get(k));
                    System.out.println("phrases extracted from easy set for class " + k+" ("+labelTranslator.toExtLabel(k)+"):");
                    System.out.println(goodPhrases);
                    blackList.addAll(goodPhrases);


                    for (String ngram:goodTerms){
                        int featureIndex = featureMappers.nextAvailable();
                        SearchResponse response = index.match(index.getBodyField(),
                                ngram,trainIdTranslator.getAllExtIds(), MatchQueryBuilder.Operator.AND);
                        for (SearchHit hit: response.getHits().getHits()){
                            String indexId = hit.getId();
                            int algorithmId = trainIdTranslator.toIntId(indexId);
                            float score = hit.getScore();
                            dataSet.setFeatureValue(algorithmId, featureIndex,score);
                        }

                        NumericalFeatureMapper mapper = NumericalFeatureMapper.getBuilder().
                                setFeatureIndex(featureIndex).setFeatureName(ngram).
                                setSource("matching_score").build();
                        featureMappers.addMapper(mapper);
                        blackList.add(ngram);
                    }

                    for (String phrase:goodPhrases){
                        int featureIndex = featureMappers.nextAvailable();
                        SearchResponse response = index.matchPhrase(index.getBodyField(),
                                phrase,trainIdTranslator.getAllExtIds(), 0);
                        for (SearchHit hit: response.getHits().getHits()){
                            String indexId = hit.getId();
                            int algorithmId = trainIdTranslator.toIntId(indexId);
                            float score = hit.getScore();
                            dataSet.setFeatureValue(algorithmId, featureIndex,score);
                        }

                        NumericalFeatureMapper mapper = NumericalFeatureMapper.getBuilder().
                                setFeatureIndex(featureIndex).setFeatureName(phrase).
                                setSource("matching_score").build();
                        featureMappers.addMapper(mapper);
                    }
                }
            }

            /**
             * focus set
             */
            if (shouldExtractFeatures&&config.getBoolean("extraction.fromHardSet")){
                //generate focus set
                FocusSet focusSet = new FocusSet(numClasses);
                for (int k=0;k<numClasses;k++){
                    double[] gradient = boosting.getGradients(k);
                    Comparator<Pair<Integer,Double>> comparator = Comparator.comparing(Pair::getSecond);
                    List<Integer> hardExamples = IntStream.range(0,gradient.length)
                            .mapToObj(i -> new Pair<>(i,gradient[i]))
                            .filter(pair -> pair.getSecond()>0)
                            .sorted(comparator.reversed())
                            .limit(numDocsToSelect)
                            .map(Pair::getFirst)
                            .collect(Collectors.toList());
                    for(Integer doc: hardExamples){
                        focusSet.add(doc,k);
                    }
                }

                List<Integer> validationSet = focusSet.getAll();

                for (int k=0;k<numClasses;k++){

                    double[] allGradients = boosting.getGradients(k);
                    List<Double> gradientsForValidation = validationSet.stream()
                            .map(i -> allGradients[i]).collect(Collectors.toList());

                    List<String> goodTerms = null;

                    goodTerms = tfidfSplitExtractor.getGoodTerms(focusSet,
                            validationSet,
                            blackList, k, gradientsForValidation);

                    seedsForAllClasses.get(k).addAll(goodTerms);

                    List<String> focusSetIndexIds = focusSet.getDataClassK(k)
                            .parallelStream().map(trainIdTranslator::toExtId)
                            .collect(Collectors.toList());
                    System.out.println("hard set for class " +k+ "("+labelTranslator.toExtLabel(k)+ "):");
                    System.out.println(focusSetIndexIds.toString());
                    System.out.println("terms extracted from hard set for class " + k+" ("+labelTranslator.toExtLabel(k)+"):");
                    System.out.println(goodTerms);

                    //phrases
                    System.out.println("seeds for class " +k+ "("+labelTranslator.toExtLabel(k)+ "):");
                    System.out.println(seedsForAllClasses.get(k));
                    List<String> goodPhrases = phraseSplitExtractor.getGoodPhrases(focusSet,validationSet,blackList,k,
                            gradientsForValidation,seedsForAllClasses.get(k));
                    System.out.println("phrases extracted from hard set for class " + k+" ("+labelTranslator.toExtLabel(k)+"):");
                    System.out.println(goodPhrases);
                    blackList.addAll(goodPhrases);

                    for (String ngram:goodTerms){
                        int featureIndex = featureMappers.nextAvailable();
                        SearchResponse response = index.match(index.getBodyField(),
                                ngram,trainIdTranslator.getAllExtIds(), MatchQueryBuilder.Operator.AND);
                        for (SearchHit hit: response.getHits().getHits()){
                            String indexId = hit.getId();
                            int algorithmId = trainIdTranslator.toIntId(indexId);
                            float score = hit.getScore();
                            dataSet.setFeatureValue(algorithmId, featureIndex,score);
                        }

                        NumericalFeatureMapper mapper = NumericalFeatureMapper.getBuilder().
                                setFeatureIndex(featureIndex).setFeatureName(ngram).
                                setSource("matching_score").build();
                        featureMappers.addMapper(mapper);
                        blackList.add(ngram);
                    }

                    for (String phrase:goodPhrases){
                        int featureIndex = featureMappers.nextAvailable();
                        SearchResponse response = index.matchPhrase(index.getBodyField(),
                                phrase,trainIdTranslator.getAllExtIds(), 0);
                        for (SearchHit hit: response.getHits().getHits()){
                            String indexId = hit.getId();
                            int algorithmId = trainIdTranslator.toIntId(indexId);
                            float score = hit.getScore();
                            dataSet.setFeatureValue(algorithmId, featureIndex,score);
                        }

                        NumericalFeatureMapper mapper = NumericalFeatureMapper.getBuilder().
                                setFeatureIndex(featureIndex).setFeatureName(phrase).
                                setSource("matching_score").build();
                        featureMappers.addMapper(mapper);
                    }
                }
            }

            int[] activeFeatures = IntStream.range(0, featureMappers.getTotalDim()).toArray();
            boosting.setActiveFeatures(activeFeatures);
            boosting.fitRegressors();
        }

        File serializedModel =  new File(archive,modelName);
        if (!overwriteModels && serializedModel.exists()){
            throw new RuntimeException(serializedModel.getAbsolutePath()+"already exists");
        }

        boosting.serialize(serializedModel);
        System.out.println("model saved to "+serializedModel.getAbsolutePath());
        System.out.println("accuracy on training set = "+ Accuracy.accuracy(boosting,
                dataSet));
        System.out.println("time spent = "+stopWatch);

    }

    static DFStats loadDFStats(MultiLabelIndex index, IdTranslator trainIdTranslator, LabelTranslator labelTranslator) throws IOException {
        DFStats dfStats = new DFStats(labelTranslator.getNumClasses());
        String[] trainIds = trainIdTranslator.getAllExtIds();
        dfStats.update(index,labelTranslator,trainIds);
        dfStats.sort();
        return dfStats;
    }

    static void build(Config config, MultiLabelIndex index) throws Exception{
        int numDocsInIndex = index.getNumDocs();
        String[] trainIndexIds = sampleTrain(config,index);
        System.out.println("number of training documents = "+trainIndexIds.length);
        IdTranslator trainIdTranslator = loadIdTranslator(trainIndexIds);
        FeatureMappers featureMappers = new FeatureMappers();
        LabelTranslator trainLabelTranslator = loadTrainLabelTranslator(index,trainIndexIds);
        if (config.getBoolean("useInitialFeatures")){
            addInitialFeatures(config,index,featureMappers,trainIndexIds);
        }

        MultiLabelClfDataSet trainDataSet = loadTrainData(config,index,featureMappers, trainIdTranslator, trainLabelTranslator);

        trainModel(config,trainDataSet,featureMappers,index, trainIdTranslator);

        //only keep used columns
        List<Integer> columns = IntStream.range(0,featureMappers.getTotalDim()).mapToObj(i-> i)
                .collect(Collectors.toList());
        MultiLabelClfDataSet trimmedTrainDataSet = DataSetUtil.trim(trainDataSet,columns);
        DataSetUtil.setFeatureMappers(trimmedTrainDataSet,featureMappers);
        saveDataSet(config, trimmedTrainDataSet, config.getString("archive.trainingSet"));
        if (config.getBoolean("archive.dumpFields")){
            dumpTrainFields(config, index, trainIdTranslator);
        }

        String[] testIndexIds = sampleTest(numDocsInIndex,trainIndexIds);
        IdTranslator testIdTranslator = loadIdTranslator(testIndexIds);
        LabelTranslator testLabelTranslator = loadTestLabelTranslator(index,testIndexIds,trainLabelTranslator);

        MultiLabelClfDataSet testDataSet = loadTestData(config,index,featureMappers,testIdTranslator,testLabelTranslator);
        DataSetUtil.setFeatureMappers(testDataSet,featureMappers);
        saveDataSet(config, testDataSet, config.getString("archive.testSet"));
        if (config.getBoolean("archive.dumpFields")){
            dumpTestFields(config, index, testIdTranslator);
        }
    }

    static LabelTranslator loadTrainLabelTranslator(MultiLabelIndex index, String[] trainIndexIds) throws Exception{

        Set<String> extLabelSet = new HashSet<>();

        for (String i: trainIndexIds){
            List<String> extLabel = index.getExtMultiLabel(i);
            extLabelSet.addAll(extLabel);
        }

        LabelTranslator labelTranslator = new LabelTranslator(extLabelSet);

        System.out.println("there are "+labelTranslator.getNumClasses()+" classes in the training set.");
        System.out.println(labelTranslator);
        return labelTranslator;
    }

    static LabelTranslator loadTestLabelTranslator(MultiLabelIndex index, String[] testIndexIds, LabelTranslator trainLabelTranslator){
        List<String> extLabels = new ArrayList<>();
        for (int i=0;i<trainLabelTranslator.getNumClasses();i++){
            extLabels.add(trainLabelTranslator.toExtLabel(i));
        }

        Set<String> testExtLabelSet = new HashSet<>();
        for (String i: testIndexIds){
            List<String> extLabel = index.getExtMultiLabel(i);
            testExtLabelSet.addAll(extLabel);
        }

        testExtLabelSet.removeAll(extLabels);
        for (String extLabel: testExtLabelSet){
            extLabels.add(extLabel);
        }

        return new LabelTranslator(extLabels);

    }

}
