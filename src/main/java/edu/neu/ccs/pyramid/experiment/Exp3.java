package edu.neu.ccs.pyramid.experiment;

import edu.neu.ccs.pyramid.classification.boosting.lktb.LKTBConfig;
import edu.neu.ccs.pyramid.classification.boosting.lktb.LKTreeBoost;
import edu.neu.ccs.pyramid.configuration.Config;
import edu.neu.ccs.pyramid.dataset.*;
import edu.neu.ccs.pyramid.elasticsearch.ESIndex;
import edu.neu.ccs.pyramid.elasticsearch.ESIndexBuilder;
import edu.neu.ccs.pyramid.dataset.IdTranslator;
import edu.neu.ccs.pyramid.eval.Accuracy;
import edu.neu.ccs.pyramid.feature.*;
import edu.neu.ccs.pyramid.feature_extraction.*;
import edu.neu.ccs.pyramid.util.Pair;
import edu.neu.ccs.pyramid.util.Sampling;
import org.apache.commons.lang3.time.StopWatch;
import org.elasticsearch.action.search.SearchResponse;
import org.elasticsearch.index.query.MatchQueryBuilder;
import org.elasticsearch.search.SearchHit;

import java.io.File;
import java.io.IOException;
import java.util.*;
import java.util.regex.Pattern;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

/**
 *  feature extraction using elasticsearch,
 *  can start with initial features in ESindex
 * can use default train/test split or random split
 * follow golden_features, exp57
 * Created by chengli on 9/7/14.
 */
public class Exp3 {

    public static void main(String[] args) throws Exception{
        if (args.length !=1){
            throw new IllegalArgumentException("please specify the config file");
        }

        Config config = new Config(args[0]);
        System.out.println(config);

        ESIndex index = loadIndex(config);

        if (config.getBoolean("train")){
            train(config,index);
        }

        index.close();

    }

    static void train(Config config, ESIndex index) throws Exception{
        int numDocsInIndex = index.getNumDocs();
        String[] trainIndexIds = sampleTrain(config,index);
        System.out.println("number of training documents = "+trainIndexIds.length);
        IdTranslator trainIdTranslator = loadIdTranslator(trainIndexIds);
        FeatureMappers featureMappers = new FeatureMappers();
        Map<Integer, String> labelMap = intToExtLabel(config,index);
        if (config.getBoolean("useInitialFeatures")){
            addInitialFeatures(config,index,featureMappers,trainIndexIds);
        }

        ClfDataSet trainDataSet = loadTrainData(config,index,featureMappers, trainIdTranslator, labelMap);
        System.out.println("in training set :");
        showDistribution(config,trainDataSet,labelMap);

        trainModel(config,trainDataSet,featureMappers,index, trainIdTranslator);

        //only keep used columns
        ClfDataSet trimmedTrainDataSet = DataSetUtil.trim(trainDataSet,featureMappers.getTotalDim());
        DataSetUtil.setFeatureMappers(trimmedTrainDataSet,featureMappers);
        saveDataSet(config, trimmedTrainDataSet, config.getString("archive.trainingSet"));

        String[] testIndexIds = sampleTest(numDocsInIndex,trainIndexIds);
        IdTranslator testIdTranslator = loadIdTranslator(testIndexIds);

        ClfDataSet testDataSet = loadTestData(config,index,featureMappers,testIdTranslator,labelMap);
        DataSetUtil.setFeatureMappers(testDataSet,featureMappers);
        saveDataSet(config, testDataSet, config.getString("archive.testSet"));
    }



    static ESIndex loadIndex(Config config) throws Exception{
        ESIndexBuilder builder = ESIndexBuilder.builder()
                .setIndexName(config.getString("index.indexName"))
                .setClusterName(config.getString("index.clusterName"))
                .setClientType(config.getString("index.clientType"))
                .setLabelField(config.getString("index.labelField"))
                .setExtLabelField(config.getString("index.extLabelField"))
                .setDocumentType(config.getString("index.documentType"));
        if (config.getString("index.clientType").equals("transport")){
            String[] hosts = config.getString("index.hosts").split(Pattern.quote(","));
            String[] ports = config.getString("index.ports").split(Pattern.quote(","));
            builder.addHostsAndPorts(hosts,ports);
        }
        ESIndex index = builder.build();
        System.out.println("index loaded");
        System.out.println("there are "+index.getNumDocs()+" documents in the index.");
//        for (int i=0;i<index.getNumDocs();i++){
//            System.out.println(i);
//            System.out.println(index.getLabel(""+i));
//        }
        return index;
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
    static ClfDataSet loadData(Config config, ESIndex index,
                               FeatureMappers featureMappers,
                               IdTranslator idTranslator, int totalDim,
                               Map<Integer,String> labelMap) throws Exception{
        int numDataPoints = idTranslator.numData();
        int numClasses = config.getInt("numClasses");
        ClfDataSet dataSet;
        if(config.getBoolean("featureMatrix.sparse")){
            dataSet= new SparseClfDataSet(numDataPoints,totalDim,numClasses);
        } else {
            dataSet= new DenseClfDataSet(numDataPoints,totalDim,numClasses);
        }
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
        DataSetUtil.setIdTranslator(dataSet,idTranslator);
        DataSetUtil.setExtLabels(dataSet,labelMap);
        return dataSet;
    }

    static ClfDataSet loadTrainData(Config config, ESIndex index, FeatureMappers featureMappers,
                                    IdTranslator idTranslator, Map<Integer,String> labelMap) throws Exception{
        System.out.println("creating training set");


        //todo fix

        int totalDim = config.getInt("maxNumColumns");
        System.out.println("allocating "+totalDim+" columns for training set");
        ClfDataSet dataSet = loadData(config,index,featureMappers,idTranslator,totalDim,labelMap);
        System.out.println("training set created");
        return dataSet;
    }

    static ClfDataSet loadTestData(Config config, ESIndex index,
                                   FeatureMappers featureMappers, IdTranslator idTranslator,
                                   Map<Integer,String> labelMap) throws Exception{
        System.out.println("creating test set");

        int totalDim = featureMappers.getTotalDim();

        ClfDataSet dataSet = loadData(config,index,featureMappers,idTranslator,totalDim,labelMap);
        System.out.println("test set created");
        return dataSet;
    }

    static void trainModel(Config config, ClfDataSet dataSet, FeatureMappers featureMappers,
                           ESIndex index, IdTranslator trainIdTranslator) throws Exception{
        String archive = config.getString("archive.folder");
        int numIterations = config.getInt("train.numIterations");
        int numClasses = config.getInt("numClasses");
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

        Map<Integer,String> labelMap = dataSet.getSetting().getLabelMap();

        StopWatch stopWatch = new StopWatch();
        stopWatch.start();

        System.out.println("training model ");

        LKTBConfig trainConfig = new LKTBConfig.Builder(dataSet)
                .learningRate(learningRate).minDataPerLeaf(trainMinDataPerLeaf)
                .numLeaves(numLeaves)
                .build();
        LKTreeBoost lkTreeBoost = new LKTreeBoost(numClasses);
        lkTreeBoost.setPriorProbs(dataSet);
        lkTreeBoost.setTrainConfig(trainConfig);

        TermSplitExtractor splitExtractor = new TermSplitExtractor(index, trainIdTranslator,
                numNgramsToExtract)
                .setMinDataPerLeaf(config.getInt("extraction.splitExtractor.minDataPerLeaf"));

        TermTfidfExtractor tfidfExtractor = new TermTfidfExtractor(index,trainIdTranslator,
                numNgramsToExtract).
                setMinDf(config.getInt("extraction.tfidfExtractor.minDf"));

        TermTfidfSplitExtractor tfidfSplitExtractor = new TermTfidfSplitExtractor(index,
                trainIdTranslator,numNgramsToExtract).
                setMinDf(config.getInt("extraction.tfidfSplitExtractor.minDf")).
                setNumSurvivors(config.getInt("extraction.tfidfSplitExtractor.numSurvivors")).
                setMinDataPerLeaf(config.getInt("extraction.tfidfSplitExtractor.minDataPerLeaf"));

        PhraseSplitExtractor phraseSplitExtractor = new PhraseSplitExtractor(index,trainIdTranslator)
                .setMinDataPerLeaf(config.getInt("extraction.phraseSplitExtractor.minDataPerLeaf"))
                .setMinDf(config.getInt("extraction.phraseSplitExtractor.minDf"))
                .setTopN(config.getInt("extraction.phraseSplitExtractor.topN"));

        DFStats dfStats = loadDFStats(config,index,trainIdTranslator);
        List<Set<String>> seedsForAllClasses = new ArrayList<>();
        for (int i=0;i<numClasses;i++){
            //todo parameters
            Set<String> set = new HashSet<>();
            set.addAll(dfStats.getSortedTerms(i,50,100));
            seedsForAllClasses.add(set);
        }

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
            lkTreeBoost.calGradients();

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
                    double[] gradient = lkTreeBoost.getGradient(k);
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
                    double[] allGradients = lkTreeBoost.getGradient(k);
                    List<Double> gradientsForValidation = validationSet.stream()
                            .map(i -> allGradients[i]).collect(Collectors.toList());

                    List<String> goodTerms = null;
                    if(config.getString("extraction.Extractor").equalsIgnoreCase("splitExtractor")){
                        goodTerms = splitExtractor.getGoodTerms(focusSet,
                                validationSet,
                                blackList, k, gradientsForValidation);
                    } else if (config.getString("extraction.Extractor").equalsIgnoreCase("tfidfExtractor")){
                        goodTerms = tfidfExtractor.getGoodTerms(focusSet,blackList,k);
                    } else if (config.getString("extraction.Extractor").equalsIgnoreCase("tfidfSplitExtractor")){
                        goodTerms = tfidfSplitExtractor.getGoodTerms(focusSet,
                                validationSet,
                                blackList, k, gradientsForValidation);
                    } else {
                        throw new RuntimeException("ngram extractor is not specified correctly");
                    }

                    seedsForAllClasses.get(k).addAll(goodTerms);

                    List<String> focusSetIndexIds = focusSet.getDataClassK(k)
                            .parallelStream().map(trainIdTranslator::toExtId)
                            .collect(Collectors.toList());
                    System.out.println("easy set for class " +k+ "("+labelMap.get(k)+ "):");
                    System.out.println(focusSetIndexIds.toString());
                    System.out.println("terms extracted from easy set for class " + k+" ("+labelMap.get(k)+"):");
                    System.out.println(goodTerms);




                    //phrases
                    System.out.println("seeds for class " +k+ "("+labelMap.get(k)+ "):");
                    System.out.println(seedsForAllClasses.get(k));
                    List<String> goodPhrases = phraseSplitExtractor.getGoodPhrases(focusSet,validationSet,blackList,k,
                            gradientsForValidation,seedsForAllClasses.get(k));
                    System.out.println("phrases extracted from easy set for class " + k+" ("+labelMap.get(k)+"):");
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
                    double[] gradient = lkTreeBoost.getGradient(k);
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

                    double[] allGradients = lkTreeBoost.getGradient(k);
                    List<Double> gradientsForValidation = validationSet.stream()
                            .map(i -> allGradients[i]).collect(Collectors.toList());

                    List<String> goodTerms = null;
                    if(config.getString("extraction.Extractor").equalsIgnoreCase("splitExtractor")){
                        goodTerms = splitExtractor.getGoodTerms(focusSet,
                                validationSet,
                                blackList, k, gradientsForValidation);
                    } else if (config.getString("extraction.Extractor").equalsIgnoreCase("tfidfExtractor")){
                        goodTerms = tfidfExtractor.getGoodTerms(focusSet,blackList,k);
                    } else if (config.getString("extraction.Extractor").equalsIgnoreCase("tfidfSplitExtractor")){
                        goodTerms = tfidfSplitExtractor.getGoodTerms(focusSet,
                                validationSet,
                                blackList, k, gradientsForValidation);
                    } else {
                        throw new RuntimeException("ngram extractor is not specified correctly");
                    }
                    seedsForAllClasses.get(k).addAll(goodTerms);

                    List<String> focusSetIndexIds = focusSet.getDataClassK(k)
                            .parallelStream().map(trainIdTranslator::toExtId)
                            .collect(Collectors.toList());
                    System.out.println("hard set for class " +k+ "("+labelMap.get(k)+ "):");
                    System.out.println(focusSetIndexIds.toString());
                    System.out.println("terms extracted from hard set for class " + k+" ("+labelMap.get(k)+"):");
                    System.out.println(goodTerms);

                    //phrases
                    System.out.println("seeds for class " +k+ "("+labelMap.get(k)+ "):");
                    System.out.println(seedsForAllClasses.get(k));
                    List<String> goodPhrases = phraseSplitExtractor.getGoodPhrases(focusSet,validationSet,blackList,k,
                            gradientsForValidation,seedsForAllClasses.get(k));
                    System.out.println("phrases extracted from hard set for class " + k+" ("+labelMap.get(k)+"):");
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
            lkTreeBoost.setActiveFeatures(activeFeatures);
            lkTreeBoost.fitRegressors();
        }

        File serializedModel =  new File(archive,modelName);
        if (!overwriteModels && serializedModel.exists()){
            throw new RuntimeException(serializedModel.getAbsolutePath()+"already exists");
        }

        lkTreeBoost.serialize(serializedModel);
        System.out.println("model saved to "+serializedModel.getAbsolutePath());
        System.out.println("accuracy on training set = "+ Accuracy.accuracy(lkTreeBoost,
                dataSet));
        System.out.println("time spent = "+stopWatch);

    }

    static String[] sampleTrain(Config config, ESIndex index){
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
            double trainPercentage = config.getDouble("split.random.trainPercentage");
            int[] labels = new int[numDocsInIndex];
            for (int i=0;i<labels.length;i++){
                labels[i] = index.getLabel(""+i);
            }
            List<Integer> sample = Sampling.stratified(labels, trainPercentage);
            trainIds = new String[sample.size()];
            for (int i=0;i<trainIds.length;i++){
                trainIds[i] = ""+sample.get(i);
            }
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

    static Map<Integer, String> intToExtLabel(Config config, ESIndex index) throws Exception{
        int numClasses = config.getInt("numClasses");
        int numDocs = index.getNumDocs();
        Map<Integer, String> map = new HashMap<>(numClasses);
        for (int i=0;i<numDocs;i++){
            int intLabel = index.getLabel(""+i);
            String extLabel = index.getExtLabel("" + i);
            map.put(intLabel,extLabel);
        }
        return map;
    }

    /**
     *
     * @param config
     * @param index
     * @param featureMappers to be updated
     * @param ids pull features from train ids
     * @throws Exception
     */
    static void addInitialFeatures(Config config, ESIndex index,
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

    static void showDistribution(Config config, ClfDataSet dataSet, Map<Integer, String> labelMap){
        int numClasses = config.getInt("numClasses");
        int[] counts = new int[numClasses];
        int[] labels = dataSet.getLabels();
        for (int i=0;i<dataSet.getNumDataPoints();i++){
            int label = labels[i];
            counts[label] += 1;

        }
        System.out.println("label distribution:");
        for (int i=0;i<numClasses;i++){
            System.out.print(i+"("+labelMap.get(i)+"):"+counts[i]+", ");
        }
        System.out.println("");
    }

    static void saveDataSet(Config config, ClfDataSet dataSet, String name) throws Exception{
        String archive = config.getString("archive.folder");
        File dataFile = new File(archive,name);
        TRECFormat.save(dataSet, dataFile);
        DataSetUtil.dumpDataSettings(dataSet,new File(dataFile,"data_settings.txt"));
        DataSetUtil.dumpFeatureSettings(dataSet,new File(dataFile,"feature_settings.txt"));
        System.out.println("data set saved to "+dataFile.getAbsolutePath());
    }

    static DFStats loadDFStats(Config config, ESIndex index, IdTranslator trainIdTranslator) throws IOException {
        DFStats dfStats = new DFStats(config.getInt("numClasses"));
        String[] trainIds = trainIdTranslator.getAllExtIds();
        dfStats.update(index,trainIds);
        dfStats.sort();
        return dfStats;
    }


}
