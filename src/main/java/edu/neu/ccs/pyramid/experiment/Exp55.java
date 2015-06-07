//package edu.neu.ccs.pyramid.experiment;
//
//import edu.neu.ccs.pyramid.classification.logistic_regression.LogisticLoss;
//import edu.neu.ccs.pyramid.classification.logistic_regression.LogisticRegression;
//import edu.neu.ccs.pyramid.configuration.Config;
//import edu.neu.ccs.pyramid.dataset.*;
//import edu.neu.ccs.pyramid.elasticsearch.FeatureLoader;
//import edu.neu.ccs.pyramid.elasticsearch.SingleLabelIndex;
//import edu.neu.ccs.pyramid.eval.Accuracy;
//import edu.neu.ccs.pyramid.feature.*;
//import edu.neu.ccs.pyramid.feature_extraction.*;
//import edu.neu.ccs.pyramid.optimization.LBFGS;
//import edu.neu.ccs.pyramid.util.Pair;
//import edu.neu.ccs.pyramid.util.Sampling;
//import edu.neu.ccs.pyramid.util.SetUtil;
//import org.apache.commons.io.FileUtils;
//import org.apache.commons.io.IOUtils;
//import org.apache.commons.lang3.time.StopWatch;
//import org.elasticsearch.action.search.SearchResponse;
//import org.elasticsearch.search.SearchHit;
//
//import java.io.BufferedWriter;
//import java.io.File;
//import java.io.FileReader;
//import java.io.FileWriter;
//import java.util.*;
//import java.util.concurrent.ConcurrentHashMap;
//import java.util.regex.Pattern;
//import java.util.stream.Collectors;
//import java.util.stream.IntStream;
//
///**
// * feature extraction by logistic regression,
// * start with specified featureList
// * extract both unigrams and ngrams
// * dynamic seeds
// * Created by chengli on 1/19/15.
// */
//public class Exp55 {
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
//        LabelTranslator labelTranslator = loadLabelTranslator(config, index);
//        FeatureList featureList = new FeatureList();
//        loadInitialFeaturesFromFile(config,featureList);
//        Set<String> duplidate = loadDuplicate(config);
//        String[] trainIndexIds = sampleTrain(config,index,duplidate);
//        IdTranslator trainIdTranslator = loadIdTranslator(trainIndexIds);
//
//        ClfDataSet trainDataSet = loadTrainSet(config, index, featureList,trainIdTranslator,labelTranslator);
//        System.out.println("in training set :");
//        showDistribution(config,trainDataSet,trainDataSet.getLabelTranslator());
//
//        trainModel(config, trainDataSet, featureList,
//                index, trainDataSet.getIdTranslator());
//
//        //only keep used columns
//        ClfDataSet trimmedTrainDataSet = DataSetUtil.sampleFeatures(trainDataSet, featureList.size());
//        trimmedTrainDataSet.setFeatureList(featureList);
//        saveDataSet(config, trimmedTrainDataSet, config.getString("archive.trainingSet"));
//        if (config.getBoolean("archive.dumpFields")){
//            dumpTrainFeatures(config,index,trimmedTrainDataSet.getIdTranslator());
//        }
//
//        String[] testIndexIds = sampleTest(config,index);
//        IdTranslator testIdTranslator = loadIdTranslator(testIndexIds);
//
//        ClfDataSet testDataSet = loadTestSet(config, index, featureList,testIdTranslator,labelTranslator);
//        testDataSet.setFeatureList(featureList);
//        saveDataSet(config, testDataSet, config.getString("archive.testSet"));
//        if (config.getBoolean("archive.dumpFields")){
//            dumpTestFeatures(config,index,testDataSet.getIdTranslator());
//        }
//
////        ClfDataSet validDataSet = loadValidSet(config, index, featureMappers);
////        saveDataSet(config, validDataSet, config.getString("archive.validSet"));
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
//                                   FeatureList featureList, IdTranslator idTranslator,
//                                   LabelTranslator labelTranslator) throws Exception{
//        System.out.println("creating test set");
//
//        int totalDim = featureList.size();
//
//        ClfDataSet dataSet = loadData(config,index,featureList,idTranslator,totalDim,labelTranslator);
//        System.out.println("test set created");
//        return dataSet;
//    }
//
//    static void trainModel(Config config, ClfDataSet dataSet, FeatureList featureList,
//                           SingleLabelIndex index, IdTranslator trainIdTranslator) throws Exception{
//        String archive = config.getString("archive.folder");
//        File archiveFolder = new File(archive);
//        archiveFolder.mkdirs();
//        int numIterations = config.getInt("train.numIterations");
//        int numClasses = dataSet.getNumClasses();
//
//        String modelName = config.getString("archive.model");
//
//
//
//        LabelTranslator labelTranslator = dataSet.getLabelTranslator();
//
//        StopWatch stopWatch = new StopWatch();
//        stopWatch.start();
//
//        System.out.println("training model ");
//
//        int[] classCounts = new int[numClasses];
//        IntStream.range(0,dataSet.getNumDataPoints()).map(i-> dataSet.getLabels()[i])
//                .forEach(label -> classCounts[label]+=1);
//
//
//        LogisticRegression logisticRegression = new LogisticRegression(numClasses,dataSet.getNumFeatures());
//        logisticRegression.setFeatureExtraction(true);
//        LogisticLoss logisticLoss = new LogisticLoss(logisticRegression,
//                dataSet,config.getDouble("train.gaussianPriorVariance"));
//        LBFGS lbfgs;
//
//
//        int[] topNs = new int[numClasses];
//        for (int k=0;k<numClasses;k++){
//            topNs[k] = (classCounts[k]*config.getInt("extraction.topN")/dataSet.getNumDataPoints());
//        }
//
//        System.out.println("ngrams extraction from each class = "+Arrays.toString(topNs));
//
//
//        Set<String> blackList = new HashSet<>();
//
//        //add initial unigrams to blacklist
//        for (Feature feature: featureList.getAll()){
//            if (feature instanceof Ngram){
//                int length = ((Ngram) feature).getN();
//                if (length==1){
//                    String ngram = ((Ngram) feature).getNgram();
//                    blackList.add(ngram);
//                }
//
//            }
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
//        int focuseSetSize = config.getInt("extraction.focusSet.size");
//        int[] numDocsPerFocusSet = new int[numClasses];
//        double focusPercentage = ((double)focuseSetSize)/dataSet.getNumDataPoints();
//        for (int k=0;k<numClasses;k++){
//            numDocsPerFocusSet[k] = (int)(classCounts[k]*focusPercentage/numFocusSets);
//        }
//
//        System.out.println("focus set sizes = "+Arrays.toString(numDocsPerFocusSet));
//
//        Set<String> validationSets = config.getStrings("extraction.validationSet.type").stream().collect(Collectors.toSet());
//        int numValidationSets = validationSets.size();
//
//        //todo
//        int validationSize = config.getInt("extraction.validationSet.size");
//        double validationPercentage = ((double)validationSize)/dataSet.getNumDataPoints();
//        int[] numDocsPerValidationSet = new int[numClasses];
//
//        for (int k=0;k<numClasses;k++){
//            numDocsPerValidationSet[k] = (int)(classCounts[k]*validationPercentage/numValidationSets);
//        }
//        System.out.println("validation set sizes = "+Arrays.toString(numDocsPerValidationSet));
//
//        //todo
//        int numSeeds = config.getInt("extraction.numSeeds");
//
//        File statsFile = new File(config.getString("archive.folder"),"stats");
//        BufferedWriter statsWriter = new BufferedWriter(new FileWriter(statsFile));
//
//        statsWriter.write("initially");
//        statsWriter.write(",");
//        statsWriter.write("number of features = " + featureList.size());
//        statsWriter.newLine();
//
//        List<List<String>> setCoverDocs = new ArrayList<>();
//        if (!config.getString("input.setCoverDocs").equals("")){
//            FileReader fileReader = new FileReader(config.getString("input.setCoverDocs"));
//            List<String> lines = IOUtils.readLines(fileReader);
//            for (String line: lines){
//                List<String> list = new ArrayList<>();
//                for (String doc: line.split(",")){
//                    list.add(doc.trim());
//                }
//                setCoverDocs.add(list);
//            }
//        }
//
//        List<List<Integer>> setCoverDocAlgorithmIds = setCoverDocs.parallelStream()
//                .map(list -> list.stream().map(trainIdTranslator::toIntId).collect(Collectors.toList()))
//                .collect(Collectors.toList());
//
//
//        for (int iteration=0;iteration<numIterations;iteration++) {
//            System.out.println("iteration " + iteration);
//
//            logisticLoss.refresh();
//            System.out.println("loss at the start of iteration " + iteration + " = " + logisticLoss.getValue());
//            lbfgs = new LBFGS(logisticLoss);
//            lbfgs.optimize();
//            System.out.println("loss after the optimization " + iteration + " = " + logisticLoss.getValue());
//
//
//            boolean condition1 = (featureList.size()
//                    + config.getInt("extraction.topN")
//                    < dataSet.getNumFeatures());
//
//
//            boolean shouldExtractFeatures = condition1;
//
//            if (!shouldExtractFeatures) {
//                if (!condition1) {
//                    System.out.println("we have reached the max number of columns " +
//                            "and will not extract new featureList");
//                    break;
//                }
//            }
//
//
//
//            if (shouldExtractFeatures) {
//                FocusSet focusSet = new FocusSet(numClasses);
//                focusSetProducer.setGradientMatrix(logisticLoss.getGradientMatrix());
//                focusSetProducer.setProbabilityMatrix(logisticLoss.getProbabilityMatrix());
//
//
//                validationSetProducer.setGradientMatrix(logisticLoss.getGradientMatrix());
//                validationSetProducer.setProbabilityMatrix(logisticLoss.getProbabilityMatrix());
//
//                for (int k = 0; k < numClasses; k++) {
//
//                    if (focusSets.contains("easy")){
//                        Set<Integer> easySet = focusSetProducer.produceEasyOnes(k, numDocsPerFocusSet[k]);
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
//                                    + "/" + numDocsPerFocusSet[k] + " common documents in the easy set.");
//                        }
//                    }
//
//
//                    if (focusSets.contains("hard")){
//                        Set<Integer> hardSet = focusSetProducer.produceHardOnes(k, numDocsPerFocusSet[k]);
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
//                                    + "/" + numDocsPerFocusSet[k] + " common documents in the hard set.");
//                        }
//                    }
//
//                    if (focusSets.contains("uncertain")){
//                        Set<Integer> uncertainSet = focusSetProducer.produceUncertainOnes(k, numDocsPerFocusSet[k]);
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
//                                    + "/" + numDocsPerFocusSet[k] + " common documents in the uncertain set.");
//                        }
//                    }
//
//                    if (focusSets.contains("random")){
//                        Set<Integer> randomSet = focusSetProducer.produceRandomOnes(k, numDocsPerFocusSet[k]);
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
//                    if (focusSets.contains("setCover")){
//                        focusSetProducer.setDesiredOrders(setCoverDocAlgorithmIds);
//                        Set<Integer> set = focusSetProducer.produceDesiredOnes(k, numDocsPerFocusSet[k]);
//                        List<String> indexIds = set
//                                .parallelStream().map(trainIdTranslator::toExtId).sorted()
//                                .collect(Collectors.toList());
//                        System.out.println("set cover set for class " + k + "(" + labelTranslator.toExtLabel(k) + "):");
//                        System.out.println(indexIds.toString());
//                        for (Integer dataPoint : set) {
//                            focusSet.add(dataPoint, k);
//                        }
//
//                    }
//
//
//                }
//
//
//                System.out.println("focus set = "+focusSet.getAll());
//
//                List<Integer> validationSet = new ArrayList<>();
//                for (int k = 0; k < numClasses; k++){
//                    if (validationSets.contains("easy")){
//                        validationSet.addAll(validationSetProducer.produceEasyOnes(k,numDocsPerValidationSet[k]));
//                    }
//
//                    if (validationSets.contains("hard")){
//                        validationSet.addAll(validationSetProducer.produceHardOnes(k,numDocsPerValidationSet[k]));
//                    }
//
//                    if (validationSets.contains("uncertain")){
//                        validationSet.addAll(validationSetProducer.produceUncertainOnes(k,numDocsPerValidationSet[k]));
//                    }
//
//                    if (validationSets.contains("random")){
//                        validationSet.addAll(validationSetProducer.produceRandomOnes(k,numDocsPerValidationSet[k]));
//                    }
//
//                }
//
//                TermTfidfSplitExtractor termExtractor = new TermTfidfSplitExtractor(index,
//                        trainIdTranslator,validationSet).
//                        setMinDf(config.getInt("extraction.termExtractor.minDf")).
//                        setNumSurvivors(config.getInt("extraction.termExtractor.numSurvivors")).
//                        setMinDataPerLeaf(config.getInt("extraction.termExtractor.minDataPerLeaf"));
//
//                PhraseSplitExtractor phraseSplitExtractor = new PhraseSplitExtractor(index,trainIdTranslator,validationSet)
//                        .setMinDataPerLeaf(config.getInt("extraction.phraseExtractor.minDataPerLeaf"))
//                        .setMinDf(config.getInt("extraction.phraseExtractor.minDf"))
//                        .setLengthLimit(config.getInt("extraction.phraseExtractor.maxN"));
//
//                MixedSplitExtractor mixedSplitExtractor = new MixedSplitExtractor(termExtractor,phraseSplitExtractor);
//
//
//                for (int k = 0; k < numClasses; k++) {
//                    double[] allGradients = logisticLoss.getGradientMatrix().getGradientsForClass(k);
//                    List<Double> gradientsForValidation = validationSet.stream()
//                            .map(i -> allGradients[i]).collect(Collectors.toList());
//
//                    //phrases
//                    List<String> goodPhrases = mixedSplitExtractor.getGoodNgrams(focusSet, blackList, k,
//                            gradientsForValidation, numSeeds,topNs[k]);
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
//                        int featureIndex = featureList.nextAvailable();
//                        for (SearchHit hit : response.getHits().getHits()) {
//                            String indexId = hit.getId();
//                            int algorithmId = trainIdTranslator.toIntId(indexId);
//                            float score = hit.getScore();
//                            dataSet.setFeatureValue(algorithmId, featureIndex, score);
//                        }
//
//                        Ngram ngram = new Ngram();
//                        //todo
//                        ngram.setField("body");
//                        ngram.setSlop(0);
//                        ngram.setName(phrase);
//                        ngram.setNgram(phrase);
//                        ngram.getSettings().put("source","matching_score");
//                        featureList.add(ngram);
//                    }
//
//                }
//
//
//                statsWriter.write("iteration = " + iteration);
//                statsWriter.write(",");
//                statsWriter.write("focus set = " + focusSet.getAll());
//                statsWriter.write(",");
//                statsWriter.write("number of features = " + featureList.size());
//                statsWriter.newLine();
//
//
//            }
//
//        }
//
//        statsWriter.close();
//        File serializedModel =  new File(archive,modelName);
//        logisticRegression.serialize(serializedModel);
//        System.out.println("model saved to "+serializedModel.getAbsolutePath());
//        System.out.println("accuracy on training set = "+ Accuracy.accuracy(logisticRegression,
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
//    static ClfDataSet loadTrainSet(Config config, SingleLabelIndex index, FeatureList featureList,
//                                    IdTranslator idTranslator, LabelTranslator labelTranslator) throws Exception{
//        System.out.println("creating training set");
//        int totalDim = config.getInt("featureMatrix.maxNumColumns");
//        System.out.println("allocating "+totalDim+" columns for training set");
//        ClfDataSet dataSet = loadData(config,index,featureList,idTranslator,totalDim,labelTranslator);
//        System.out.println("training set created");
//        return dataSet;
//    }
//
//    //todo
//    /**
//     * assuming unigram for now
//     */
//    static void loadInitialFeaturesFromFile(Config config, FeatureList featureList) throws Exception{
//
//        File initialFeatureFile = new File(config.getString("input.initialFeatureFile"));
//        String[] line = FileUtils.readLines(initialFeatureFile).get(0).split(",");
//        List<String> unigrams = Arrays.stream(line).collect(Collectors.toList());
//        System.out.println("initial featureList:");
//        System.out.println(unigrams);
//        for (String unigram: unigrams){
//            Ngram ngram = new Ngram();
//            ngram.setNgram(unigram);
//            ngram.setSlop(0);
//            ngram.setName(unigram);
//            ngram.setField("body");
//            ngram.getSettings().put("source", "matching_score");
//            featureList.add(ngram);
//        }
//    }
//
//    static String[] sampleTrain(Config config, SingleLabelIndex index, Set<String> duplicate){
//        int numDocsInIndex = index.getNumDocs();
//        String[] ids = null;
//
//        String splitField = config.getString("index.splitField");
//        List<String> train = IntStream.range(0, numDocsInIndex).parallel()
//                .filter(i -> index.getStringField("" + i, splitField).
//                        equalsIgnoreCase("train")).
//                        mapToObj(i -> "" + i).filter(id -> !duplicate.contains(id)).collect(Collectors.toList());
//        ids = train.toArray(new String[train.size()]);
//        return ids;
//    }
//
//    static String[] sampleTest(Config config, SingleLabelIndex index){
//        int numDocsInIndex = index.getNumDocs();
//        String[] ids = null;
//
//        String splitField = config.getString("index.splitField");
//        ids = IntStream.range(0, numDocsInIndex).parallel().
//                filter(i -> index.getStringField("" + i, splitField).
//                        equalsIgnoreCase("test")).
//                mapToObj(i -> "" + i).collect(Collectors.toList()).
//                toArray(new String[0]);
//        return ids;
//    }
//
//    static IdTranslator loadIdTranslator(String[] indexIds) throws Exception{
//        IdTranslator idTranslator = new IdTranslator();
//        for (int i=0;i<indexIds.length;i++){
//            idTranslator.addData(i,""+indexIds[i]);
//        }
//        return idTranslator;
//    }
//
//    static ClfDataSet loadData(Config config, SingleLabelIndex index,
//                               FeatureList featureList,
//                               IdTranslator idTranslator, int totalDim,
//                               LabelTranslator labelTranslator) throws Exception{
//        int numDataPoints = idTranslator.numData();
//        int numClasses = config.getInt("numClasses");
//        ClfDataSet dataSet = ClfDataSetBuilder.getBuilder()
//                .numDataPoints(numDataPoints).numFeatures(totalDim)
//                .numClasses(numClasses).dense(!config.getBoolean("featureMatrix.sparse"))
//                .missingValue(config.getBoolean("featureMatrix.missingValue"))
//                .build();
//
//        IntStream.range(0,numDataPoints).parallel()
//                .forEach(i -> {
//                    String dataIndexId = idTranslator.toExtId(i);
//                    int label = index.getLabel(dataIndexId);
//                    dataSet.setLabel(i,label);
//                });
//
//        FeatureLoader.loadFeatures(index, dataSet, featureList, idTranslator);
//
//        dataSet.setIdTranslator(idTranslator);
//        dataSet.setLabelTranslator(labelTranslator);
//        return dataSet;
//    }
//
//    static LabelTranslator loadLabelTranslator(Config config, SingleLabelIndex index) throws Exception{
//        System.out.println("loading label translator...");
//        int numClasses = config.getInt("numClasses");
//        int numDocs = index.getNumDocs();
//        Map<Integer, String> map = new ConcurrentHashMap<>();
//        while(map.size()<numClasses){
//            int i = Sampling.intUniform(0,numDocs-1);
//            int intLabel = index.getLabel(""+i);
//            String extLabel = index.getExtLabel("" + i);
//            map.put(intLabel,extLabel);
//        }
//        System.out.println("loaded");
//        return new LabelTranslator(map);
//    }
//
//    static Set<String> loadDuplicate(Config config) throws Exception{
//        File file = new File(config.getString("input.duplicate"));
//        String[] strArr = FileUtils.readFileToString(file).split(",");
//        Set<String> set = new HashSet<>();
//        Arrays.stream(strArr).forEach(set::add);
//        return set;
//    }
//}
