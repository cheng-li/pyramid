//package edu.neu.ccs.pyramid.experiment;
//
//import edu.neu.ccs.pyramid.active_learning.BestVsSecond;
//import edu.neu.ccs.pyramid.classification.boosting.lktb.LKTBConfig;
//import edu.neu.ccs.pyramid.classification.boosting.lktb.LKTreeBoost;
//import edu.neu.ccs.pyramid.configuration.Config;
//import edu.neu.ccs.pyramid.dataset.*;
//import edu.neu.ccs.pyramid.elasticsearch.SingleLabelIndex;
//import edu.neu.ccs.pyramid.eval.Accuracy;
//import edu.neu.ccs.pyramid.feature.FeatureMappers;
//import edu.neu.ccs.pyramid.feature.NumericalFeatureMapper;
//import edu.neu.ccs.pyramid.feature_extraction.*;
//import edu.neu.ccs.pyramid.util.Pair;
//import edu.neu.ccs.pyramid.util.Sampling;
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
// * start with specified features
// * extract both unigrams and ngrams
// * Created by chengli on 12/30/14.
// */
//public class Exp48 {
//    public static void main(String[] args) throws Exception{
//        if (args.length !=1){
//            throw new IllegalArgumentException("please specify the config file");
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
//    static void train(Config config, SingleLabelIndex index) throws Exception{
//        FeatureMappers featureMappers = new FeatureMappers();
//        loadInitialFeaturesFromFile(config,featureMappers);
//
//        ClfDataSet trainDataSet = loadTrainSet(config, index, featureMappers);
//        System.out.println("in training set :");
//        showDistribution(config,trainDataSet,trainDataSet.getSetting().getLabelTranslator());
//
//        trainModel(config,trainDataSet,featureMappers,
//                index, trainDataSet.getSetting().getIdTranslator());
//
//        //only keep used columns
//        ClfDataSet trimmedTrainDataSet = DataSetUtil.trim(trainDataSet, featureMappers.getTotalDim());
//        DataSetUtil.setFeatureMappers(trimmedTrainDataSet,featureMappers);
//        saveDataSet(config, trimmedTrainDataSet, config.getString("archive.trainingSet"));
//        if (config.getBoolean("archive.dumpFields")){
//            dumpTrainFeatures(config,index,trimmedTrainDataSet.getSetting().getIdTranslator());
//        }
//
//        ClfDataSet testDataSet = loadTestSet(config, index, featureMappers);
//        saveDataSet(config, testDataSet, config.getString("archive.testSet"));
//        if (config.getBoolean("archive.dumpFields")){
//            dumpTestFeatures(config,index,testDataSet.getSetting().getIdTranslator());
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
//        DataSetUtil.setLabelTranslator(dataSet,initialDataSet.getSetting().getLabelTranslator());
//        DataSetUtil.setIdTranslator(dataSet,initialDataSet.getSetting().getIdTranslator());
//        dataSet.getSetting().setFeatureMappers(trainFeatureMappers);
//
//        IdTranslator idTranslator = initialDataSet.getSetting().getIdTranslator();
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
//        DataSetUtil.setLabelTranslator(dataSet,initialDataSet.getSetting().getLabelTranslator());
//        DataSetUtil.setIdTranslator(dataSet,initialDataSet.getSetting().getIdTranslator());
//        dataSet.getSetting().setFeatureMappers(trainFeatureMappers);
//
//        IdTranslator idTranslator = initialDataSet.getSetting().getIdTranslator();
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
//        int numDocsToSelect = config.getInt("extraction.numDocsToSelect");
//
//
//        LabelTranslator labelTranslator = dataSet.getSetting().getLabelTranslator();
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
//        LinkedList<Set<Integer>> easySets = new LinkedList<>();
//        LinkedList<Set<Integer>> hardSets = new LinkedList<>();
//        LinkedList<Set<Integer>> uncertainSets = new LinkedList<>();
//
//        for (int iteration=0;iteration<numIterations;iteration++){
//            System.out.println("iteration "+iteration);
//
//            int[] activeFeatures = IntStream.range(0,featureMappers.getTotalDim()).toArray();
//            boosting.setActiveFeatures(activeFeatures);
//            System.out.println("running boosting");
//            for (int i=0;i<config.getInt("train.boostingRounds");i++){
//                boosting.boostOneRound();
//            }
//            System.out.println("done");
//
//            boolean condition1 = (featureMappers.getTotalDim()
//                    +config.getInt("extraction.topN")*numClasses*3
//                    <dataSet.getNumFeatures());
//
//
//            boolean shouldExtractFeatures = condition1;
//
//            if (!shouldExtractFeatures){
//                if (!condition1){
//                    System.out.println("we have reached the max number of columns " +
//                            "and will not extract new features");
//                }
//            }
//
//
//
//            /**
//             * from easy set
//             */
//            if (shouldExtractFeatures&&config.getBoolean("extraction.fromEasySet")){
//                //generate easy set
//                FocusSet focusSet = new FocusSet(numClasses);
//                for (int k=0;k<numClasses;k++){
//                    double[] gradient = boosting.getGradient(k);
//                    Comparator<Pair<Integer,Double>> comparator = Comparator.comparing(Pair::getSecond);
//                    List<Integer> easyExamples = IntStream.range(0, gradient.length)
//                            .mapToObj(i -> new Pair<>(i,gradient[i]))
//                            .filter(pair -> pair.getSecond()>0)
//                            .sorted(comparator)
//                            .limit(numDocsToSelect)
//                            .map(Pair::getFirst)
//                            .collect(Collectors.toList());
//                    for(Integer doc: easyExamples){
//                        focusSet.add(doc,k);
//                    }
//                }
//
//                easySets.add(new HashSet<Integer>(focusSet.getAll()));
//                if (easySets.size()>2){
//                    easySets.remove();
//                }
//                if (iteration>=1){
//                    int common = SetUtil.intersect(easySets.getFirst(), easySets.getLast()).size();
//                    System.out.println("between iterations "+(iteration-1)+" and "+iteration+", there are "+common
//                            +"/"+numDocsToSelect*numClasses+" common documents in the easy set.");
//                }
//
//                List<Integer> validationSet;
//                if (config.getString("extraction.validation.fashion").equals("fixed")){
//                    validationSet = focusSet.getAll();
//                } else {
//                    List<Integer> allIndices = IntStream.range(0,dataSet.getNumDataPoints()).mapToObj(i->i)
//                            .collect(Collectors.toList());
//                    validationSet = Sampling.sampleByPercentage(allIndices, config.getDouble("extraction.validation.random.percentage"));
//                }
//
//
//                for (int k=0;k<numClasses;k++){
//                    double[] allGradients = boosting.getGradient(k);
//                    List<Double> gradientsForValidation = validationSet.stream()
//                            .map(i -> allGradients[i]).collect(Collectors.toList());
//
//
//
//                    List<String> focusSetIndexIds = focusSet.getDataClassK(k)
//                            .parallelStream().map(trainIdTranslator::toExtId)
//                            .collect(Collectors.toList());
//                    System.out.println("easy set for class " +k+ "("+labelTranslator.toExtLabel(k)+ "):");
//                    System.out.println(focusSetIndexIds.toString());
//
//                    //phrases
//                    System.out.println("seeds for class " +k+ "("+labelTranslator.toExtLabel(k)+ "):");
//                    System.out.println(seedsForAllClasses.get(k));
//                    List<String> goodPhrases = mixedSplitExtractor.getGoodNgrams(focusSet,validationSet,blackList,k,
//                            gradientsForValidation,seedsForAllClasses.get(k));
//                    System.out.println("phrases extracted from easy set for class " + k+" ("+labelTranslator.toExtLabel(k)+"):");
//                    System.out.println(goodPhrases);
//                    blackList.addAll(goodPhrases);
//
//
//                    List<Pair<String,SearchResponse>> searchResponseList = goodPhrases.stream().parallel()
//                            .map(phrase -> new Pair<>(phrase,index.matchPhrase(index.getBodyField(),
//                                    phrase,trainIdTranslator.getAllExtIds(), 0)))
//                            .collect(Collectors.toList());
//
//                    for (Pair<String,SearchResponse> pair: searchResponseList){
//                        String phrase = pair.getFirst();
//                        SearchResponse response = pair.getSecond();
//                        int featureIndex = featureMappers.nextAvailable();
//                        for (SearchHit hit: response.getHits().getHits()){
//                            String indexId = hit.getId();
//                            int algorithmId = trainIdTranslator.toIntId(indexId);
//                            float score = hit.getScore();
//                            dataSet.setFeatureValue(algorithmId, featureIndex,score);
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
//            /**
//             * hard set
//             */
//            if (shouldExtractFeatures&&config.getBoolean("extraction.fromHardSet")){
//                //generate focus set
//                FocusSet focusSet = new FocusSet(numClasses);
//                for (int k=0;k<numClasses;k++){
//                    double[] gradient = boosting.getGradient(k);
//                    Comparator<Pair<Integer,Double>> comparator = Comparator.comparing(Pair::getSecond);
//                    List<Integer> hardExamples = IntStream.range(0,gradient.length)
//                            .mapToObj(i -> new Pair<>(i,gradient[i]))
//                            .filter(pair -> pair.getSecond()>0)
//                            .sorted(comparator.reversed())
//                            .limit(numDocsToSelect)
//                            .map(Pair::getFirst)
//                            .collect(Collectors.toList());
//                    for(Integer doc: hardExamples){
//                        focusSet.add(doc,k);
//                    }
//                }
//
//                hardSets.add(new HashSet<Integer>(focusSet.getAll()));
//                if (hardSets.size()>2){
//                    hardSets.remove();
//                }
//                if (iteration>=1){
//                    int common = SetUtil.intersect(hardSets.getFirst(),hardSets.getLast()).size();
//                    System.out.println("between iterations "+(iteration-1)+" and "+iteration+", there are "+common
//                            +"/"+numDocsToSelect*numClasses+" common documents in the hard set.");
//                }
//
//                List<Integer> validationSet;
//                if (config.getString("extraction.validation.fashion").equals("fixed")){
//                    validationSet = focusSet.getAll();
//                } else {
//                    List<Integer> allIndices = IntStream.range(0,dataSet.getNumDataPoints()).mapToObj(i->i)
//                            .collect(Collectors.toList());
//                    validationSet = Sampling.sampleByPercentage(allIndices,config.getDouble("extraction.validation.random.percentage"));
//                }
//
//                for (int k=0;k<numClasses;k++){
//
//                    double[] allGradients = boosting.getGradient(k);
//                    List<Double> gradientsForValidation = validationSet.stream()
//                            .map(i -> allGradients[i]).collect(Collectors.toList());
//
//
//
//                    List<String> focusSetIndexIds = focusSet.getDataClassK(k)
//                            .parallelStream().map(trainIdTranslator::toExtId)
//                            .collect(Collectors.toList());
//                    System.out.println("hard set for class " +k+ "("+labelTranslator.toExtLabel(k)+ "):");
//                    System.out.println(focusSetIndexIds.toString());
//
//
//                    //phrases
//                    System.out.println("seeds for class " +k+ "("+labelTranslator.toExtLabel(k)+ "):");
//                    System.out.println(seedsForAllClasses.get(k));
//                    List<String> goodPhrases = mixedSplitExtractor.getGoodNgrams(focusSet, validationSet, blackList, k,
//                            gradientsForValidation, seedsForAllClasses.get(k));
//                    System.out.println("phrases extracted from hard set for class " + k+" ("+labelTranslator.toExtLabel(k)+"):");
//                    System.out.println(goodPhrases);
//                    blackList.addAll(goodPhrases);
//
//
//                    List<Pair<String,SearchResponse>> searchResponseList = goodPhrases.stream().parallel()
//                            .map(phrase -> new Pair<>(phrase,index.matchPhrase(index.getBodyField(),
//                                    phrase,trainIdTranslator.getAllExtIds(), 0)))
//                            .collect(Collectors.toList());
//
//                    for (Pair<String,SearchResponse> pair: searchResponseList){
//                        String phrase = pair.getFirst();
//                        SearchResponse response = pair.getSecond();
//                        int featureIndex = featureMappers.nextAvailable();
//                        for (SearchHit hit: response.getHits().getHits()){
//                            String indexId = hit.getId();
//                            int algorithmId = trainIdTranslator.toIntId(indexId);
//                            float score = hit.getScore();
//                            dataSet.setFeatureValue(algorithmId, featureIndex,score);
//                        }
//
//                        NumericalFeatureMapper mapper = NumericalFeatureMapper.getBuilder().
//                                setFeatureIndex(featureIndex).setFeatureName(phrase).
//                                setSource("matching_score").build();
//                        featureMappers.addMapper(mapper);
//                    }
//                }
//            }
//
//            /**
//             * uncertain set
//             */
//            if (shouldExtractFeatures&&config.getBoolean("extraction.fromUncertainSet")){
//                //generate focus set
//                FocusSet focusSet = new FocusSet(numClasses);
//                Comparator<Pair<Integer,BestVsSecond>> comparator = Comparator.comparing(pair -> pair.getSecond().getDifference());
//                List<Integer> examples = IntStream.range(0,dataSet.getNumDataPoints())
//                        .mapToObj(i -> new Pair<>(i,new BestVsSecond(boosting.getClassProbs(i))))
//                        .filter(pair -> (dataSet.getLabels()[pair.getFirst()]==pair.getSecond().getBestClass())
//                                ||(dataSet.getLabels()[pair.getFirst()]==pair.getSecond().getSecondClass()))
//                        .sorted(comparator).map(Pair::getFirst)
//                        .limit(numClasses*numDocsToSelect).collect(Collectors.toList());
//                for (Integer dataPoint: examples){
//                    int label = dataSet.getLabels()[dataPoint];
//                    focusSet.add(dataPoint,label);
//                }
//
//                uncertainSets.add(new HashSet<Integer>(focusSet.getAll()));
//                if (uncertainSets.size()>2){
//                    uncertainSets.remove();
//                }
//                if (iteration>=1){
//                    int common = SetUtil.intersect(uncertainSets.getFirst(),uncertainSets.getLast()).size();
//                    System.out.println("between iterations "+(iteration-1)+" and "+iteration+", there are "+common
//                            +"/"+numDocsToSelect*numClasses+" common documents in the uncertain set.");
//                }
//
//                List<Integer> validationSet;
//                if (config.getString("extraction.validation.fashion").equals("fixed")){
//                    validationSet = focusSet.getAll();
//                } else {
//                    List<Integer> allIndices = IntStream.range(0,dataSet.getNumDataPoints()).mapToObj(i->i)
//                            .collect(Collectors.toList());
//                    validationSet = Sampling.sampleByPercentage(allIndices,config.getDouble("extraction.validation.random.percentage"));
//                }
//
//                for (int k=0;k<numClasses;k++){
//
//                    double[] allGradients = boosting.getGradient(k);
//                    List<Double> gradientsForValidation = validationSet.stream()
//                            .map(i -> allGradients[i]).collect(Collectors.toList());
//
//
//                    List<String> focusSetIndexIds = focusSet.getDataClassK(k)
//                            .parallelStream().map(trainIdTranslator::toExtId)
//                            .collect(Collectors.toList());
//                    System.out.println("uncertain set for class " +k+ "("+labelTranslator.toExtLabel(k)+ "):");
//                    System.out.println(focusSetIndexIds.toString());
//
//                    //phrases
//                    System.out.println("seeds for class " +k+ "("+labelTranslator.toExtLabel(k)+ "):");
//                    System.out.println(seedsForAllClasses.get(k));
//                    List<String> goodPhrases = mixedSplitExtractor.getGoodNgrams(focusSet, validationSet, blackList, k,
//                            gradientsForValidation, seedsForAllClasses.get(k));
//                    System.out.println("phrases extracted from uncertain set for class " + k+" ("+labelTranslator.toExtLabel(k)+"):");
//                    System.out.println(goodPhrases);
//                    blackList.addAll(goodPhrases);
//
//
//                    List<Pair<String,SearchResponse>> searchResponseList = goodPhrases.stream().parallel()
//                            .map(phrase -> new Pair<>(phrase,index.matchPhrase(index.getBodyField(),
//                                    phrase,trainIdTranslator.getAllExtIds(), 0)))
//                            .collect(Collectors.toList());
//
//                    for (Pair<String,SearchResponse> pair: searchResponseList){
//                        String phrase = pair.getFirst();
//                        SearchResponse response = pair.getSecond();
//                        int featureIndex = featureMappers.nextAvailable();
//                        for (SearchHit hit: response.getHits().getHits()){
//                            String indexId = hit.getId();
//                            int algorithmId = trainIdTranslator.toIntId(indexId);
//                            float score = hit.getScore();
//                            dataSet.setFeatureValue(algorithmId, featureIndex,score);
//                        }
//
//                        NumericalFeatureMapper mapper = NumericalFeatureMapper.getBuilder().
//                                setFeatureIndex(featureIndex).setFeatureName(phrase).
//                                setSource("matching_score").build();
//                        featureMappers.addMapper(mapper);
//                    }
//                }
//            }
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
//            List<String> seedForClass = seedExtractor.getSeeds(k,config.getInt("seeds.initial.size"));
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
//        DataSetUtil.setLabelTranslator(dataSet, initialDataSet.getSetting().getLabelTranslator());
//        DataSetUtil.setIdTranslator(dataSet, initialDataSet.getSetting().getIdTranslator());
//
//        String[] dataIndexIds = new String[dataSet.getNumDataPoints()];
//        for (int i=0;i<dataSet.getNumDataPoints();i++){
//            dataIndexIds[i] = dataSet.getDataPointSetting(i).getExtId();
//        }
//
//        IdTranslator idTranslator = initialDataSet.getSetting().getIdTranslator();
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
//        System.out.println("initial features:");
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
