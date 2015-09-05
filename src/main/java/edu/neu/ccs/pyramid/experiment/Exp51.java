//package edu.neu.ccs.pyramid.experiment;
//
//
//import edu.neu.ccs.pyramid.classification.boosting.lktb.LKTBConfig;
//import edu.neu.ccs.pyramid.classification.boosting.lktb.LKTreeBoost;
//import edu.neu.ccs.pyramid.configuration.Config;
//import edu.neu.ccs.pyramid.dataset.*;
//import edu.neu.ccs.pyramid.elasticsearch.SingleLabelIndex;
//import edu.neu.ccs.pyramid.eval.Accuracy;
//import edu.neu.ccs.pyramid.feature.FeatureMappers;
//import edu.neu.ccs.pyramid.feature.NumericalFeatureMapper;
//import edu.neu.ccs.pyramid.feature_extraction.*;
//
//import edu.neu.ccs.pyramid.util.Pair;
//import edu.neu.ccs.pyramid.util.SetUtil;
//import org.apache.commons.io.FileUtils;
//import org.apache.commons.lang3.time.StopWatch;
//import org.elasticsearch.action.search.SearchResponse;
//import org.elasticsearch.search.SearchHit;
//
//import java.io.BufferedWriter;
//import java.io.File;
//import java.io.FileWriter;
//import java.util.*;
//import java.util.regex.Pattern;
//import java.util.stream.Collectors;
//import java.util.stream.IntStream;
//
///**
// * obsolete
// * feature extraction by boosting, with train/valid/test split
// * start with specified featureList
// * extract both unigrams and ngrams
// * use focus set with promotion
// * Created by chengli on 1/8/15.
// */
//public class Exp51 {
//    public static void main(String[] args) throws Exception{
//        if (args.length !=1){
//            throw new IllegalArgumentException("Please specify a properties file.");
//        }
//
//        Config config = new Config(args[0]);
//        System.out.println(config);
//
//        SingleLabelIndex index = loadIndex(config);
//
//
//        train(config,index);
//
//
//        index.close();
//
//    }
//
//    public static void mainFromConfig(Config config) throws Exception{
//
//        SingleLabelIndex index = loadIndex(config);
//
//
//        train(config,index);
//
//
//        index.close();
//
//    }
//
//    static void train(Config config, SingleLabelIndex index) throws Exception{
//        FeatureMappers featureMappers = new FeatureMappers();
//        loadInitialFeaturesFromFile(config,featureMappers);
//
//        ClfDataSet trainDataSet = loadTrainSet(config, index, featureMappers);
//        System.out.println("in training set :");
//        showDistribution(config,trainDataSet,trainDataSet.getSettings().getLabelTranslator());
//
//        trainModel(config,trainDataSet,featureMappers,
//                index, trainDataSet.getSettings().getIdTranslator());
//
//        //only keep used columns
//        ClfDataSet trimmedTrainDataSet = DataSetUtil.sampleFeatures(trainDataSet, featureMappers.getTotalDim());
//        DataSetUtil.setFeatureMappers(trimmedTrainDataSet,featureMappers);
//        saveDataSet(config, trimmedTrainDataSet, config.getString("archive.trainingSet"));
//        if (config.getBoolean("archive.dumpFields")){
//            dumpTrainFeatures(config,index,trimmedTrainDataSet.getSettings().getIdTranslator());
//        }
//
//        ClfDataSet testDataSet = loadTestSet(config, index, featureMappers);
//        saveDataSet(config, testDataSet, config.getString("archive.testSet"));
//        if (config.getBoolean("archive.dumpFields")){
//            dumpTestFeatures(config,index,testDataSet.getSettings().getIdTranslator());
//        }
//
//        ClfDataSet validDataSet = loadValidSet(config, index, featureMappers);
//        saveDataSet(config, validDataSet, config.getString("archive.validSet"));
//    }
//
//
//
//    static SingleLabelIndex loadIndex(Config config) throws Exception{
//        SingleLabelIndex.Builder builder = new SingleLabelIndex.Builder()
//                .setIndexName(config.getString("index.indexName"))
//                .setClusterName(config.getString("index.clusterName"))
//                .setClientType(config.getString("index.clientType"))
//                .setLabelField(config.getString("index.labelField"))
//                .setExtLabelField(config.getString("index.extLabelField"))
//                .setDocumentType(config.getString("index.documentType"));
//        if (config.getString("index.clientType").equals("transport")){
//            String[] hosts = config.getString("index.hosts").split(Pattern.quote(","));
//            String[] ports = config.getString("index.ports").split(Pattern.quote(","));
//            builder.addHostsAndPorts(hosts,ports);
//        }
//        SingleLabelIndex index = builder.build();
//        System.out.println("index loaded");
//        System.out.println("there are "+index.getNumDocs()+" documents in the index.");
////        for (int i=0;i<index.getNumDocs();i++){
////            System.out.println(i);
////            System.out.println(index.getLabel(""+i));
////        }
//        return index;
//    }
//
//
//    static ClfDataSet loadTestSet(Config config, SingleLabelIndex index,
//                                  FeatureMappers trainFeatureMappers) throws Exception{
//        File initialDataSetFile = new File(config.getString("input.initialDataSets"),"test.trec");
//        ClfDataSet initialDataSet = TRECFormat.loadClfDataSet(initialDataSetFile, DataSetType.CLF_SPARSE, true);
//        int totalDim = trainFeatureMappers.getTotalDim();
//        ClfDataSet dataSet = ClfDataSetBuilder.getBuilder()
//                .dense(false).numDataPoints(initialDataSet.getNumDataPoints())
//                .numFeatures(totalDim)
//                .numClasses(initialDataSet.getNumClasses())
//                .missingValue(initialDataSet.hasMissingValue())
//                .build();
//
//        for (int i=0;i<initialDataSet.getNumDataPoints();i++){
//            dataSet.setLabel(i,initialDataSet.getLabels()[i]);
//        }
//
//        DataSetUtil.setLabelTranslator(dataSet,initialDataSet.getSettings().getLabelTranslator());
//        DataSetUtil.setIdTranslator(dataSet,initialDataSet.getSettings().getIdTranslator());
//        dataSet.getSettings().setFeatureMappers(trainFeatureMappers);
//
//        IdTranslator idTranslator = initialDataSet.getSettings().getIdTranslator();
//        String[] dataIndexIds = new String[dataSet.getNumDataPoints()];
//        for (int i=0;i<dataSet.getNumDataPoints();i++){
//            dataIndexIds[i] = dataSet.getDataPointSetting(i).getExtId();
//        }
//
//        trainFeatureMappers.getNumericalFeatureMappers().stream().parallel().
//                forEach(numericalFeatureMapper -> {
//                    String featureName = numericalFeatureMapper.getFeatureName();
//                    int featureIndex = numericalFeatureMapper.getFeatureIndex();
//
//                    SearchResponse response = null;
//
//                    //todo assume unigram, so slop doesn't matter
//                    response = index.matchPhrase(index.getBodyField(), featureName, dataIndexIds, 0);
//
//                    SearchHit[] hits = response.getHits().getHits();
//                    for (SearchHit hit: hits){
//                        String indexId = hit.getId();
//                        float score = hit.getScore();
//                        int algorithmId = idTranslator.toIntId(indexId);
//                        dataSet.setFeatureValue(algorithmId,featureIndex,score);
//                    }
//                });
//        DataSetUtil.setFeatureMappers(dataSet,trainFeatureMappers);
//        return dataSet;
//    }
//
//    static ClfDataSet loadValidSet(Config config, SingleLabelIndex index,
//                                   FeatureMappers trainFeatureMappers) throws Exception{
//        File initialDataSetFile = new File(config.getString("input.initialDataSets"),"valid.trec");
//        ClfDataSet initialDataSet = TRECFormat.loadClfDataSet(initialDataSetFile, DataSetType.CLF_SPARSE, true);
//        int totalDim = trainFeatureMappers.getTotalDim();
//        ClfDataSet dataSet = ClfDataSetBuilder.getBuilder()
//                .dense(false).numDataPoints(initialDataSet.getNumDataPoints())
//                .numFeatures(totalDim)
//                .numClasses(initialDataSet.getNumClasses())
//                .missingValue(initialDataSet.hasMissingValue())
//                .build();
//
//        for (int i=0;i<initialDataSet.getNumDataPoints();i++){
//            dataSet.setLabel(i,initialDataSet.getLabels()[i]);
//        }
//
//        DataSetUtil.setLabelTranslator(dataSet,initialDataSet.getSettings().getLabelTranslator());
//        DataSetUtil.setIdTranslator(dataSet,initialDataSet.getSettings().getIdTranslator());
//        dataSet.getSettings().setFeatureMappers(trainFeatureMappers);
//
//        IdTranslator idTranslator = initialDataSet.getSettings().getIdTranslator();
//        String[] dataIndexIds = new String[dataSet.getNumDataPoints()];
//        for (int i=0;i<dataSet.getNumDataPoints();i++){
//            dataIndexIds[i] = dataSet.getDataPointSetting(i).getExtId();
//        }
//
//        trainFeatureMappers.getNumericalFeatureMappers().stream().parallel().
//                forEach(numericalFeatureMapper -> {
//                    String featureName = numericalFeatureMapper.getFeatureName();
//                    int featureIndex = numericalFeatureMapper.getFeatureIndex();
//                    SearchResponse response = null;
//
//                    //todo assume unigram, so slop doesn't matter
//                    response = index.matchPhrase(index.getBodyField(), featureName, dataIndexIds, 0);
//
//                    SearchHit[] hits = response.getHits().getHits();
//                    for (SearchHit hit: hits){
//                        String indexId = hit.getId();
//                        float score = hit.getScore();
//                        int algorithmId = idTranslator.toIntId(indexId);
//                        dataSet.setFeatureValue(algorithmId,featureIndex,score);
//                    }
//
//                });
//
//        DataSetUtil.setFeatureMappers(dataSet,trainFeatureMappers);
//        return dataSet;
//    }
//
//    static void trainModel(Config config, ClfDataSet dataSet, FeatureMappers featureMappers,
//                           SingleLabelIndex index, IdTranslator trainIdTranslator) throws Exception{
//        String archive = config.getString("archive.folder");
//        int numIterations = config.getInt("train.numIterations");
//        int numClasses = dataSet.getNumClasses();
//
//        String modelName = config.getString("archive.model");
//        int numDocsToSelect = config.getInt("extraction.focusSet.numDocs");
//
//
//        LabelTranslator labelTranslator = dataSet.getSettings().getLabelTranslator();
//
//        StopWatch stopWatch = new StopWatch();
//        stopWatch.start();
//
//        System.out.println("training model ");
//
//        LKTBConfig trainConfig = new LKTBConfig.Builder(dataSet)
//                .learningRate(config.getDouble("train.learningRate"))
//                .minDataPerLeaf(config.getInt("train.minDataPerLeaf"))
//                .numLeaves(config.getInt("train.numLeaves"))
//                .build();
//        LKTreeBoost boosting = new LKTreeBoost(numClasses);
//        boosting.setPriorProbs(dataSet);
//        boosting.setTrainConfig(trainConfig);
//
//        TermTfidfSplitExtractor termExtractor = new TermTfidfSplitExtractor(index,
//                trainIdTranslator).
//                setMinDf(config.getInt("extraction.termExtractor.minDf")).
//                setNumSurvivors(config.getInt("extraction.termExtractor.numSurvivors")).
//                setMinDataPerLeaf(config.getInt("extraction.termExtractor.minDataPerLeaf"));
//
//        PhraseSplitExtractor phraseSplitExtractor = new PhraseSplitExtractor(index,trainIdTranslator)
//                .setMinDataPerLeaf(config.getInt("extraction.phraseExtractor.minDataPerLeaf"))
//                .setMinDf(config.getInt("extraction.phraseExtractor.minDf"));
//
//        MixedSplitExtractor mixedSplitExtractor = new MixedSplitExtractor(termExtractor,phraseSplitExtractor);
//        mixedSplitExtractor.setTopN(config.getInt("extraction.topN"));
//
//
//        List<Set<String>> seedsForAllClasses = loadInitialSeeds(config);
//
//        Set<String> blackList = new HashSet<>();
//
//        //add initial unigrams to blacklist
//        for (int i=0;i<featureMappers.getTotalDim();i++){
//            blackList.add(featureMappers.getName(i));
//        }
//
//        List<LinkedList<Set<Integer>>> easySets = new ArrayList<>();
//        for (int k=0;k<numClasses;k++){
//            easySets.add(new LinkedList<>());
//        }
//
//        List<LinkedList<Set<Integer>>> hardSets = new ArrayList<>();
//        for (int k=0;k<numClasses;k++){
//            hardSets.add(new LinkedList<>());
//        }
//
//        List<LinkedList<Set<Integer>>> uncertainSets = new ArrayList<>();
//        for (int k=0;k<numClasses;k++){
//            uncertainSets.add(new LinkedList<>());
//        }
//
//
//        FocusSetProducer focusSetProducer = new FocusSetProducer(numClasses,dataSet.getNumDataPoints());
//        focusSetProducer.setPromotion(config.getBoolean("extraction.focusSet.promotion"));
//        focusSetProducer.setLabels(numClasses,dataSet.getLabels());
//
//        FocusSetProducer validationSetProducer = new FocusSetProducer(numClasses,dataSet.getNumDataPoints());
//        validationSetProducer.setPromotion(config.getBoolean("extraction.validationSet.promotion"));
//        validationSetProducer.setLabels(numClasses,dataSet.getLabels());
//
//        Set<String> focusSets = config.getStrings("extraction.focusSet.type").stream().collect(Collectors.toSet());
//        int numFocusSets = focusSets.size();
//        int numDocsPerFocusSet = numDocsToSelect/numFocusSets;
//
//        Set<String> validationSets = config.getStrings("extraction.validationSet.type").stream().collect(Collectors.toSet());
//        int numValidationSets = validationSets.size();
//        int numDocsPerValidationSet = config.getInt("extraction.validationSet.numDocs")/numValidationSets;
//
//        for (int iteration=0;iteration<numIterations;iteration++) {
//            System.out.println("iteration " + iteration);
//
//            int[] activeFeatures = IntStream.range(0, featureMappers.getTotalDim()).toArray();
//            boosting.setActiveFeatures(activeFeatures);
//            System.out.println("running boosting");
//            for (int i=0;i<config.getInt("train.boostingRounds");i++){
//                boosting.boostOneRound();
//            }
//            System.out.println("done");
//
//
//            boolean condition1 = (featureMappers.getTotalDim()
//                    + config.getInt("extraction.topN") * numClasses
//                    < dataSet.getNumFeatures());
//
//
//            boolean shouldExtractFeatures = condition1;
//
//            if (!shouldExtractFeatures) {
//                if (!condition1) {
//                    System.out.println("we have reached the max number of columns " +
//                            "and will not extract new featureList");
//                }
//            }
//
//
//
//            if (shouldExtractFeatures) {
//                FocusSet focusSet = new FocusSet(numClasses);
//                focusSetProducer.setGradientMatrix(boosting.getGradientMatrix());
//                focusSetProducer.setProbabilityMatrix(boosting.getProbabilityMatrix());
//
//
//                validationSetProducer.setGradientMatrix(boosting.getGradientMatrix());
//                validationSetProducer.setProbabilityMatrix(boosting.getProbabilityMatrix());
//
//                for (int k = 0; k < numClasses; k++) {
//
//                    if (focusSets.contains("easy")){
//                        Set<Integer> easySet = focusSetProducer.produceEasyOnes(k, numDocsPerFocusSet);
//                        List<String> easySetIndexIds = easySet
//                                .parallelStream().map(trainIdTranslator::toExtId).sorted()
//                                .collect(Collectors.toList());
//                        System.out.println("easy set for class " + k + "(" + labelTranslator.toExtLabel(k) + "):");
//                        System.out.println(easySetIndexIds.toString());
//                        for (Integer dataPoint : easySet) {
//                            focusSet.add(dataPoint, k);
//                        }
//                        easySets.get(k).add(easySet);
//                        if (easySets.get(k).size() > 2) {
//                            easySets.get(k).remove();
//                        }
//                        if (iteration >= 1) {
//                            int commonEasy = SetUtil.intersect(easySets.get(k).getFirst(), easySets.get(k).getLast()).size();
//                            System.out.println("between iterations " + (iteration - 1) + " and " + iteration + ", there are " + commonEasy
//                                    + "/" + numDocsPerFocusSet + " common documents in the easy set.");
//                        }
//                    }
//
//
//                    if (focusSets.contains("hard")){
//                        Set<Integer> hardSet = focusSetProducer.produceHardOnes(k, numDocsPerFocusSet);
//                        List<String> hardSetIndexIds = hardSet
//                                .parallelStream().map(trainIdTranslator::toExtId).sorted()
//                                .collect(Collectors.toList());
//                        System.out.println("hard set for class " + k + "(" + labelTranslator.toExtLabel(k) + "):");
//                        System.out.println(hardSetIndexIds.toString());
//                        for (Integer dataPoint : hardSet) {
//                            focusSet.add(dataPoint, k);
//                        }
//                        hardSets.get(k).add(hardSet);
//                        if (hardSets.get(k).size() > 2) {
//                            hardSets.get(k).remove();
//                        }
//                        if (iteration >= 1) {
//                            int commonHard = SetUtil.intersect(hardSets.get(k).getFirst(), hardSets.get(k).getLast()).size();
//                            System.out.println("between iterations " + (iteration - 1) + " and " + iteration + ", there are " + commonHard
//                                    + "/" + numDocsPerFocusSet + " common documents in the hard set.");
//                        }
//                    }
//
//                    if (focusSets.contains("uncertain")){
//                        Set<Integer> uncertainSet = focusSetProducer.produceUncertainOnes(k, numDocsPerFocusSet);
//                        List<String> uncertainSetIndexIds = uncertainSet
//                                .parallelStream().map(trainIdTranslator::toExtId).sorted()
//                                .collect(Collectors.toList());
//                        System.out.println("uncertain set for class " + k + "(" + labelTranslator.toExtLabel(k) + "):");
//                        System.out.println(uncertainSetIndexIds.toString());
//                        for (Integer dataPoint : uncertainSet) {
//                            focusSet.add(dataPoint, k);
//                        }
//                        uncertainSets.get(k).add(uncertainSet);
//                        if (uncertainSets.get(k).size() > 2) {
//                            uncertainSets.get(k).remove();
//                        }
//                        if (iteration >= 1) {
//                            int commonUncertain = SetUtil.intersect(uncertainSets.get(k).getFirst(), uncertainSets.get(k).getLast()).size();
//                            System.out.println("between iterations " + (iteration - 1) + " and " + iteration + ", there are " + commonUncertain
//                                    + "/" + numDocsPerFocusSet + " common documents in the uncertain set.");
//                        }
//                    }
//
//                    if (focusSets.contains("random")){
//                        Set<Integer> randomSet = focusSetProducer.produceRandomOnes(k, numDocsPerFocusSet);
//                        List<String> randomSetIndexIds = randomSet
//                                .parallelStream().map(trainIdTranslator::toExtId).sorted()
//                                .collect(Collectors.toList());
//                        System.out.println("random set for class " + k + "(" + labelTranslator.toExtLabel(k) + "):");
//                        System.out.println(randomSetIndexIds.toString());
//                        for (Integer dataPoint : randomSet) {
//                            focusSet.add(dataPoint, k);
//                        }
//
//                    }
//
//
//                }
//
//
//                List<Integer> validationSet = new ArrayList<>();
//                for (int k = 0; k < numClasses; k++){
//                    if (validationSets.contains("easy")){
//                        validationSet.addAll(validationSetProducer.produceEasyOnes(k,numDocsPerValidationSet));
//                    }
//
//                    if (validationSets.contains("hard")){
//                        validationSet.addAll(validationSetProducer.produceHardOnes(k,numDocsPerValidationSet));
//                    }
//
//                    if (validationSets.contains("uncertain")){
//                        validationSet.addAll(validationSetProducer.produceUncertainOnes(k,numDocsPerValidationSet));
//                    }
//
//                    if (validationSets.contains("random")){
//                        validationSet.addAll(validationSetProducer.produceRandomOnes(k,numDocsPerValidationSet));
//                    }
//
//                }
//
//
//
//
//                for (int k = 0; k < numClasses; k++) {
//                    double[] allGradients = boosting.getGradient(k);
//                    List<Double> gradientsForValidation = validationSet.stream()
//                            .map(i -> allGradients[i]).collect(Collectors.toList());
//
//                    //phrases
//                    System.out.println("seeds for class " + k + "(" + labelTranslator.toExtLabel(k) + "):");
//                    System.out.println(seedsForAllClasses.get(k));
//                    List<String> goodPhrases = mixedSplitExtractor.getGoodNgrams(focusSet, validationSet, blackList, k,
//                            gradientsForValidation, seedsForAllClasses.get(k));
//                    System.out.println("phrases extracted for class " + k + " (" + labelTranslator.toExtLabel(k) + "):");
//                    System.out.println(goodPhrases);
//                    blackList.addAll(goodPhrases);
//
//
//                    List<Pair<String, SearchResponse>> searchResponseList = goodPhrases.stream().parallel()
//                            .map(phrase -> new Pair<>(phrase, index.matchPhrase(index.getBodyField(),
//                                    phrase, trainIdTranslator.getAllExtIds(), 0)))
//                            .collect(Collectors.toList());
//
//                    for (Pair<String, SearchResponse> pair : searchResponseList) {
//                        String phrase = pair.getFirst();
//                        SearchResponse response = pair.getSecond();
//                        int featureIndex = featureMappers.nextAvailable();
//                        for (SearchHit hit : response.getHits().getHits()) {
//                            String indexId = hit.getId();
//                            int algorithmId = trainIdTranslator.toIntId(indexId);
//                            float score = hit.getScore();
//                            dataSet.setFeatureValue(algorithmId, featureIndex, score);
//                        }
//
//                        NumericalFeatureMapper mapper = NumericalFeatureMapper.getBuilder().
//                                setFeatureIndex(featureIndex).setFeatureName(phrase).
//                                setSource("matching_score").build();
//                        featureMappers.addMapper(mapper);
//                    }
//
//
//                }
//            }
//
//        }
//
//        File serializedModel =  new File(archive,modelName);
//        boosting.serialize(serializedModel);
//        System.out.println("model saved to "+serializedModel.getAbsolutePath());
//        System.out.println("accuracy on training set = "+ Accuracy.accuracy(boosting,
//                dataSet));
//        System.out.println("time spent = "+stopWatch);
//
//    }
//
//
//
//    static void showDistribution(Config config, ClfDataSet dataSet, LabelTranslator labelTranslator){
//        int numClasses = dataSet.getNumClasses();
//        int[] counts = new int[numClasses];
//        int[] labels = dataSet.getLabels();
//        for (int i=0;i<dataSet.getNumDataPoints();i++){
//            int label = labels[i];
//            counts[label] += 1;
//
//        }
//        System.out.println("label distribution:");
//        for (int i=0;i<numClasses;i++){
//            System.out.print(i+"("+labelTranslator.toExtLabel(i)+"):"+counts[i]+", ");
//        }
//        System.out.println("");
//    }
//
//    static void saveDataSet(Config config, ClfDataSet dataSet, String name) throws Exception{
//        String archive = config.getString("archive.folder");
//        File dataFile = new File(archive,name);
//        TRECFormat.save(dataSet, dataFile);
//        DataSetUtil.dumpDataPointSettings(dataSet, new File(dataFile, "data_settings.txt"));
//        DataSetUtil.dumpFeatureSettings(dataSet,new File(dataFile,"feature_settings.txt"));
//        System.out.println("data set saved to "+dataFile.getAbsolutePath());
//    }
//
//
//    static void dumpTrainFeatures(Config config, SingleLabelIndex index, IdTranslator idTranslator) throws Exception{
//        String archive = config.getString("archive.folder");
//        String trecFile = new File(archive,config.getString("archive.trainingSet")).getAbsolutePath();
//        String file = new File(trecFile,"dumped_fields.txt").getAbsolutePath();
//        dumpFeatures(config,index,idTranslator,file);
//    }
//
//    static void dumpTestFeatures(Config config, SingleLabelIndex index, IdTranslator idTranslator) throws Exception{
//        String archive = config.getString("archive.folder");
//        String trecFile = new File(archive,config.getString("archive.testSet")).getAbsolutePath();
//        String file = new File(trecFile,"dumped_fields.txt").getAbsolutePath();
//        dumpFeatures(config,index,idTranslator,file);
//    }
//
//
//
//    static List<Set<String>> loadInitialSeeds(Config config) throws Exception{
//        File initialDataSetFile = new File(config.getString("input.initialDataSets"),"train.trec");
//        ClfDataSet trainSet = TRECFormat.loadClfDataSet(initialDataSetFile, DataSetType.CLF_SPARSE, true);
//        int numClasses = trainSet.getNumClasses();
//        List<Set<String>> seedsForAllClasses = new ArrayList<>();
//        for (int i=0;i<numClasses;i++){
//            seedsForAllClasses.add(new HashSet<>());
//        }
//
//        SeedExtractor seedExtractor = new SeedExtractor(trainSet);
//        for (int k=0;k<trainSet.getNumClasses();k++){
//            List<String> seedForClass = seedExtractor.getSeeds(k,config.getInt("extraction.seeds.initialSize"));
//            seedsForAllClasses.get(k).addAll(seedForClass);
//        }
//        return seedsForAllClasses;
//
//    }
//
//    static void dumpFeatures(Config config, SingleLabelIndex index, IdTranslator idTranslator, String fileName) throws Exception{
//
//        String[] fields = config.getString("archive.dumpedFields").split(",");
//        int numDocs = idTranslator.numData();
//        try(BufferedWriter bw = new BufferedWriter(new FileWriter(fileName))
//        ){
//            for (int intId=0;intId<numDocs;intId++){
//                bw.write("intId=");
//                bw.write(""+intId);
//                bw.write(",");
//                bw.write("extId=");
//                String extId = idTranslator.toExtId(intId);
//                bw.write(extId);
//                bw.write(",");
//                for (int i=0;i<fields.length;i++){
//                    String field = fields[i];
//                    bw.write(field+"=");
//                    bw.write(index.getStringField(extId,field));
//                    if (i!=fields.length-1){
//                        bw.write(",");
//                    }
//
//                }
//                bw.write("\n");
//            }
//        }
//
//    }
//
//
//    static ClfDataSet loadTrainSet(Config config, SingleLabelIndex index, FeatureMappers featureMappers) throws Exception{
//        File initialDataSetFile = new File(config.getString("input.initialDataSets"),"train.trec");
//        ClfDataSet initialDataSet = TRECFormat.loadClfDataSet(initialDataSetFile,DataSetType.CLF_SPARSE,true);
//        int totalDim = config.getInt("maxNumColumns");
//        ClfDataSet dataSet = ClfDataSetBuilder.getBuilder()
//                .dense(false).numDataPoints(initialDataSet.getNumDataPoints())
//                .numFeatures(totalDim)
//                .numClasses(initialDataSet.getNumClasses())
//                .missingValue(initialDataSet.hasMissingValue())
//                .build();
//
//
//        for (int i=0;i<initialDataSet.getNumDataPoints();i++){
//            dataSet.setLabel(i,initialDataSet.getLabels()[i]);
//        }
//
//        DataSetUtil.setLabelTranslator(dataSet, initialDataSet.getSettings().getLabelTranslator());
//        DataSetUtil.setIdTranslator(dataSet, initialDataSet.getSettings().getIdTranslator());
//
//        String[] dataIndexIds = new String[dataSet.getNumDataPoints()];
//        for (int i=0;i<dataSet.getNumDataPoints();i++){
//            dataIndexIds[i] = dataSet.getDataPointSetting(i).getExtId();
//        }
//
//        IdTranslator idTranslator = initialDataSet.getSettings().getIdTranslator();
//
//        featureMappers.getNumericalFeatureMappers().stream().parallel().
//                forEach(numericalFeatureMapper -> {
//                    String featureName = numericalFeatureMapper.getFeatureName();
//                    int featureIndex = numericalFeatureMapper.getFeatureIndex();
//                    SearchResponse response = null;
//
//                    //todo assume unigram, so slop doesn't matter
//                    response = index.matchPhrase(index.getBodyField(), featureName, dataIndexIds, 0);
//
//                    SearchHit[] hits = response.getHits().getHits();
//                    for (SearchHit hit: hits){
//                        String indexId = hit.getId();
//                        float score = hit.getScore();
//                        int algorithmId = idTranslator.toIntId(indexId);
//                        dataSet.setFeatureValue(algorithmId,featureIndex,score);
//                    }
//
//                });
//
//
//
//        return dataSet;
//    }
//
//    /**
//     * assuming unigram indices for now
//     */
//    static void loadInitialFeaturesFromFile(Config config, FeatureMappers featureMappers) throws Exception{
//        File initialDataSetFile = new File(config.getString("input.initialDataSets"),"train.trec");
//        ClfDataSet initialDataSet = TRECFormat.loadClfDataSet(initialDataSetFile, DataSetType.CLF_SPARSE, true);
//        File initialFeatureFile = new File(config.getString("input.initialFeatureFile"));
//        String[] line = FileUtils.readLines(initialFeatureFile).get(0).split(" ");
//        List<String> unigrams = Arrays.stream(line).map(Integer::parseInt)
//                .map(i-> initialDataSet.getFeatureSetting(i).getFeatureName()).collect(Collectors.toList());
//        System.out.println("initial featureList:");
//        System.out.println(unigrams);
//        for (String unigram: unigrams){
//            int featureIndex = featureMappers.nextAvailable();
//            NumericalFeatureMapper mapper = NumericalFeatureMapper.getBuilder().
//                    setFeatureIndex(featureIndex).setFeatureName(unigram).
//                    setSource("matching_score").build();
//            featureMappers.addMapper(mapper);
//        }
//    }
//}
