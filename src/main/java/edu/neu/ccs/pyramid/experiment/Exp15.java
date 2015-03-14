package edu.neu.ccs.pyramid.experiment;


import edu.neu.ccs.pyramid.configuration.Config;
import edu.neu.ccs.pyramid.dataset.*;
import edu.neu.ccs.pyramid.elasticsearch.ESIndex;
import edu.neu.ccs.pyramid.elasticsearch.FeatureLoader;
import edu.neu.ccs.pyramid.elasticsearch.MultiLabelIndex;
import edu.neu.ccs.pyramid.eval.Accuracy;
import edu.neu.ccs.pyramid.feature.*;
import edu.neu.ccs.pyramid.feature_extraction.*;
import edu.neu.ccs.pyramid.multilabel_classification.imlgb.IMLGBConfig;
import edu.neu.ccs.pyramid.multilabel_classification.imlgb.IMLGBTrainer;
import edu.neu.ccs.pyramid.multilabel_classification.imlgb.IMLGradientBoosting;
import edu.neu.ccs.pyramid.util.Pair;
import edu.neu.ccs.pyramid.util.Sampling;
import edu.neu.ccs.pyramid.util.SetUtil;
import org.apache.commons.lang3.time.StopWatch;
import org.elasticsearch.action.search.SearchResponse;
import org.elasticsearch.index.query.MatchQueryBuilder;
import org.elasticsearch.search.SearchHit;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
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

    private static boolean matchPrefixes(String name, Set<String> prefixes){
        for (String prefix: prefixes){
            if (name.startsWith(prefix)){
                return true;
            }
        }
        return false;
    }

    static void addInitialFeatures(Config config, ESIndex index, FeatureList featureList,
                                   String[] ids) throws Exception{
        String featureFieldPrefix = config.getString("index.featureFieldPrefix");
        Set<String> prefixes = Arrays.stream(featureFieldPrefix.split(",")).map(String::trim).collect(Collectors.toSet());

        Set<String> allFields = index.listAllFields();
        List<String> featureFields = allFields.stream().
                filter(field -> matchPrefixes(field,prefixes)).
                collect(Collectors.toList());
        System.out.println("all possible initial featureList:"+featureFields);

        for (String field: featureFields){
            String featureType = index.getFieldType(field);
            if (featureType.equalsIgnoreCase("string")){
                CategoricalFeatureExpander expander = new CategoricalFeatureExpander();
                expander.setStart(featureList.size());
                expander.setVariableName(field);
                expander.putSetting("source","field");
                for (String id: ids){
                    String category = index.getStringField(id, field);
                    expander.addCategory(category);
                }
                List<CategoricalFeature> group = expander.expand();
                boolean toAdd = true;
                if (config.getBoolean("categFeature.filter")){
                    double threshold = config.getDouble("categFeature.percentThreshold");
                    int numCategories = group.size();
                    if (numCategories> ids.length*threshold){
                        toAdd=false;
                        System.out.println("field "+field+" has too many categories "
                                +"("+numCategories+"), omitted.");
                    }
                }


                if(toAdd){
                    for (Feature feature: group){
                        featureList.add(feature);
                    }
                }

            } else {
                Feature feature = new Feature();
                feature.setName(field);
                feature.setIndex(featureList.size());
                feature.getSettings().put("source","field");
                featureList.add(feature);
            }
        }

    }


    static MultiLabelClfDataSet loadData(Config config, MultiLabelIndex index,
                                         FeatureList featureList,
                                         IdTranslator idTranslator, int totalDim,
                                         LabelTranslator labelTranslator) throws Exception{
        int numDataPoints = idTranslator.numData();
        int numClasses = labelTranslator.getNumClasses();
        MultiLabelClfDataSet dataSet = MLClfDataSetBuilder.getBuilder()
                .numDataPoints(numDataPoints).numFeatures(totalDim)
                .numClasses(numClasses).dense(!config.getBoolean("featureMatrix.sparse"))
                .missingValue(config.getBoolean("featureMatrix.missingValue")).build();
        for(int i=0;i<numDataPoints;i++){
            String dataIndexId = idTranslator.toExtId(i);
            List<String> extMultiLabel = index.getExtMultiLabel(dataIndexId);
            for (String extLabel: extMultiLabel){
                int intLabel = labelTranslator.toIntLabel(extLabel);
                dataSet.addLabel(i,intLabel);
            }
        }

        FeatureLoader.loadFeatures(index, dataSet, featureList, idTranslator);

        dataSet.setIdTranslator(idTranslator);
        dataSet.setLabelTranslator(labelTranslator);
        return dataSet;
    }


    static MultiLabelClfDataSet loadTrainData(Config config, MultiLabelIndex index, FeatureList featureList,
                                              IdTranslator idTranslator, LabelTranslator labelTranslator) throws Exception{
        int totalDim = config.getInt("maxNumColumns");
        System.out.println("creating training set");
        System.out.println("allocating "+totalDim+" columns for training set");
        MultiLabelClfDataSet dataSet = loadData(config,index,featureList,idTranslator,totalDim,labelTranslator);
        System.out.println("training set created");
        return dataSet;
    }

    static MultiLabelClfDataSet loadTestData(Config config, MultiLabelIndex index,
                                             FeatureList featureList, IdTranslator idTranslator,
                                             LabelTranslator labelTranslator) throws Exception{
        System.out.println("creating test set");

        int totalDim = featureList.size();

        MultiLabelClfDataSet dataSet = loadData(config,index,featureList,idTranslator,totalDim,labelTranslator);
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

        DataSetUtil.dumpDataPointSettings(dataSet, new File(dataFile, "data_settings.txt"));
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
                    bw.write(index.getListField(extId,field).toString());
                    if (i!=fields.length-1){
                        bw.write(",");
                    }

                }
                bw.write("\n");
            }
        }

    }

    static void trainModel(Config config, MultiLabelClfDataSet dataSet, FeatureList featureList,
                           MultiLabelIndex index, IdTranslator trainIdTranslator) throws Exception{
        String archive = config.getString("archive.folder");
        File archiveFolder = new File(archive);
        archiveFolder.mkdirs();
        int numIterations = config.getInt("train.numIterations");
        int numClasses = dataSet.getNumClasses();

        String modelName = config.getString("archive.model");



        LabelTranslator labelTranslator = dataSet.getLabelTranslator();

        StopWatch stopWatch = new StopWatch();
        stopWatch.start();

        System.out.println("training model ");

        int[] classCounts = DataSetUtil.getCountPerClass(dataSet);
        int totalCount = Arrays.stream(classCounts).sum();

        IMLGBConfig imlgbConfig = new IMLGBConfig.Builder(dataSet)
                .learningRate(config.getDouble("train.learningRate")).minDataPerLeaf(config.getInt("train.minDataPerLeaf"))
                .numLeaves(config.getInt("train.numLeaves"))
                .build();

        IMLGradientBoosting boosting = new IMLGradientBoosting(numClasses);
        IMLGBTrainer trainer = new IMLGBTrainer(imlgbConfig,boosting);


        int[] topNs = new int[numClasses];
        for (int k=0;k<numClasses;k++){
            topNs[k] = Math.max(1,classCounts[k]*config.getInt("extraction.topN")/totalCount);
        }

        System.out.println("number of ngrams to be extracted from each class = "+ Arrays.toString(topNs));


        Set<String> blackList = new HashSet<>();

        //add initial unigrams to blacklist

        for (Feature feature: featureList.getAll()){
            if (feature instanceof Ngram){
                int length = ((Ngram) feature).getN();
                if (length==1){
                    String ngram = ((Ngram) feature).getNgram();
                    blackList.add(ngram);
                }

            }
        }

        List<LinkedList<Set<Integer>>> easySets = new ArrayList<>();
        for (int k=0;k<numClasses;k++){
            easySets.add(new LinkedList<>());
        }

        List<LinkedList<Set<Integer>>> hardSets = new ArrayList<>();
        for (int k=0;k<numClasses;k++){
            hardSets.add(new LinkedList<>());
        }

        List<LinkedList<Set<Integer>>> uncertainSets = new ArrayList<>();
        for (int k=0;k<numClasses;k++){
            uncertainSets.add(new LinkedList<>());
        }

        List<List<Integer>> dataPerClass = DataSetUtil.labelToDataPoints(dataSet);

        FocusSetProducer focusSetProducer = new FocusSetProducer(numClasses,dataSet.getNumDataPoints());
        focusSetProducer.setPromotion(config.getBoolean("extraction.focusSet.promotion"));
        focusSetProducer.setDataPerClass(dataPerClass);

        FocusSetProducer validationSetProducer = new FocusSetProducer(numClasses,dataSet.getNumDataPoints());
        validationSetProducer.setPromotion(config.getBoolean("extraction.validationSet.promotion"));
        validationSetProducer.setDataPerClass(dataPerClass);

        Set<String> focusSets = config.getStrings("extraction.focusSet.type").stream().collect(Collectors.toSet());
        int numFocusSets = focusSets.size();
        int focusSetSize = config.getInt("extraction.focusSet.size");
        int[] numDocsPerFocusSet = new int[numClasses];
        double focusPercentage = ((double)focusSetSize)/totalCount;
        for (int k=0;k<numClasses;k++){
            numDocsPerFocusSet[k] = Math.max(1,(int)(classCounts[k]*focusPercentage/numFocusSets));
        }

        System.out.println("focus set sizes = "+Arrays.toString(numDocsPerFocusSet));

        Set<String> validationSets = config.getStrings("extraction.validationSet.type").stream().collect(Collectors.toSet());
        int numValidationSets = validationSets.size();

        //todo
        int validationSize = config.getInt("extraction.validationSet.size");
        double validationPercentage = ((double)validationSize)/totalCount;
        int[] numDocsPerValidationSet = new int[numClasses];

        for (int k=0;k<numClasses;k++){
            numDocsPerValidationSet[k] = Math.max(1,(int)(classCounts[k]*validationPercentage/numValidationSets));
        }
        System.out.println("validation set sizes = "+Arrays.toString(numDocsPerValidationSet));

        //todo
        int numSeeds = config.getInt("extraction.numSeeds");

        File statsFile = new File(config.getString("archive.folder"),"stats");
        BufferedWriter statsWriter = new BufferedWriter(new FileWriter(statsFile));

        statsWriter.write("initially");
        statsWriter.write(",");
        statsWriter.write("number of features = " + featureList.size());
        statsWriter.newLine();



        for (int iteration=0;iteration<numIterations;iteration++) {
            System.out.println("iteration " + iteration);
            int[] activeFeatures = IntStream.range(0, featureList.size()).toArray();
            trainer.setActiveFeatures(activeFeatures);
            System.out.println("running boosting");
            for (int i=0;i<config.getInt("train.boostingRounds");i++){
                trainer.iterate();
            }
            System.out.println("done");


            boolean condition1 = (featureList.size()
                    + config.getInt("extraction.topN")
                    < dataSet.getNumFeatures());


            boolean shouldExtractFeatures = condition1;

            if (!shouldExtractFeatures) {
                if (!condition1) {
                    System.out.println("we have reached the max number of columns " +
                            "and will not extract new featureList");
                    break;
                }
            }



            if (shouldExtractFeatures) {
                FocusSet focusSet = new FocusSet(numClasses);
                focusSetProducer.setGradientMatrix(trainer.getGradientMatrix());
                focusSetProducer.setProbabilityMatrix(trainer.getProbabilityMatrix());


                validationSetProducer.setGradientMatrix(trainer.getGradientMatrix());
                validationSetProducer.setProbabilityMatrix(trainer.getProbabilityMatrix());

                for (int k = 0; k < numClasses; k++) {

                    if (focusSets.contains("easy")){
                        Set<Integer> easySet = focusSetProducer.produceEasyOnes(k, numDocsPerFocusSet[k]);
                        List<String> easySetIndexIds = easySet
                                .parallelStream().map(trainIdTranslator::toExtId).sorted()
                                .collect(Collectors.toList());
                        System.out.println("easy set for class " + k + "(" + labelTranslator.toExtLabel(k) + "):");
                        System.out.println(easySetIndexIds.toString());
                        for (Integer dataPoint : easySet) {
                            focusSet.add(dataPoint, k);
                        }
                        easySets.get(k).add(easySet);
                        if (easySets.get(k).size() > 2) {
                            easySets.get(k).remove();
                        }
                        if (iteration >= 1) {
                            int commonEasy = SetUtil.intersect(easySets.get(k).getFirst(), easySets.get(k).getLast()).size();
                            System.out.println("between iterations " + (iteration - 1) + " and " + iteration + ", there are " + commonEasy
                                    + "/" + numDocsPerFocusSet[k] + " common documents in the easy set.");
                        }
                    }


                    if (focusSets.contains("hard")){
                        Set<Integer> hardSet = focusSetProducer.produceHardOnes(k, numDocsPerFocusSet[k]);
                        List<String> hardSetIndexIds = hardSet
                                .parallelStream().map(trainIdTranslator::toExtId).sorted()
                                .collect(Collectors.toList());
                        System.out.println("hard set for class " + k + "(" + labelTranslator.toExtLabel(k) + "):");
                        System.out.println(hardSetIndexIds.toString());
                        for (Integer dataPoint : hardSet) {
                            focusSet.add(dataPoint, k);
                        }
                        hardSets.get(k).add(hardSet);
                        if (hardSets.get(k).size() > 2) {
                            hardSets.get(k).remove();
                        }
                        if (iteration >= 1) {
                            int commonHard = SetUtil.intersect(hardSets.get(k).getFirst(), hardSets.get(k).getLast()).size();
                            System.out.println("between iterations " + (iteration - 1) + " and " + iteration + ", there are " + commonHard
                                    + "/" + numDocsPerFocusSet[k] + " common documents in the hard set.");
                        }
                    }

                    if (focusSets.contains("uncertain")){
                        Set<Integer> uncertainSet = focusSetProducer.produceUncertainOnes(k, numDocsPerFocusSet[k]);
                        List<String> uncertainSetIndexIds = uncertainSet
                                .parallelStream().map(trainIdTranslator::toExtId).sorted()
                                .collect(Collectors.toList());
                        System.out.println("uncertain set for class " + k + "(" + labelTranslator.toExtLabel(k) + "):");
                        System.out.println(uncertainSetIndexIds.toString());
                        for (Integer dataPoint : uncertainSet) {
                            focusSet.add(dataPoint, k);
                        }
                        uncertainSets.get(k).add(uncertainSet);
                        if (uncertainSets.get(k).size() > 2) {
                            uncertainSets.get(k).remove();
                        }
                        if (iteration >= 1) {
                            int commonUncertain = SetUtil.intersect(uncertainSets.get(k).getFirst(), uncertainSets.get(k).getLast()).size();
                            System.out.println("between iterations " + (iteration - 1) + " and " + iteration + ", there are " + commonUncertain
                                    + "/" + numDocsPerFocusSet[k] + " common documents in the uncertain set.");
                        }
                    }

                    if (focusSets.contains("random")){
                        Set<Integer> randomSet = focusSetProducer.produceRandomOnes(k, numDocsPerFocusSet[k]);
                        List<String> randomSetIndexIds = randomSet
                                .parallelStream().map(trainIdTranslator::toExtId).sorted()
                                .collect(Collectors.toList());
                        System.out.println("random set for class " + k + "(" + labelTranslator.toExtLabel(k) + "):");
                        System.out.println(randomSetIndexIds.toString());
                        for (Integer dataPoint : randomSet) {
                            focusSet.add(dataPoint, k);
                        }

                    }


                }


                System.out.println("focus set = "+focusSet.getAll());

                List<Integer> validationSet = new ArrayList<>();
                for (int k = 0; k < numClasses; k++){
                    if (validationSets.contains("easy")){
                        validationSet.addAll(validationSetProducer.produceEasyOnes(k,numDocsPerValidationSet[k]));
                    }

                    if (validationSets.contains("hard")){
                        validationSet.addAll(validationSetProducer.produceHardOnes(k,numDocsPerValidationSet[k]));
                    }

                    if (validationSets.contains("uncertain")){
                        validationSet.addAll(validationSetProducer.produceUncertainOnes(k,numDocsPerValidationSet[k]));
                    }

                    if (validationSets.contains("random")){
                        validationSet.addAll(validationSetProducer.produceRandomOnes(k,numDocsPerValidationSet[k]));
                    }

                }

                TermTfidfSplitExtractor termExtractor = new TermTfidfSplitExtractor(index,
                        trainIdTranslator,validationSet).
                        setMinDf(config.getInt("extraction.termExtractor.minDf")).
                        setNumSurvivors(config.getInt("extraction.termExtractor.numSurvivors")).
                        setMinDataPerLeaf(config.getInt("extraction.termExtractor.minDataPerLeaf"));

                PhraseSplitExtractor phraseSplitExtractor = new PhraseSplitExtractor(index,trainIdTranslator,validationSet)
                        .setMinDataPerLeaf(config.getInt("extraction.phraseExtractor.minDataPerLeaf"))
                        .setMinDf(config.getInt("extraction.phraseExtractor.minDf"))
                        .setLengthLimit(config.getInt("extraction.phraseExtractor.maxN"));

                MixedSplitExtractor mixedSplitExtractor = new MixedSplitExtractor(termExtractor,phraseSplitExtractor);


                for (int k = 0; k < numClasses; k++) {
                    double[] allGradients = trainer.getGradientMatrix().getGradientsForClass(k);
                    List<Double> gradientsForValidation = validationSet.stream()
                            .map(i -> allGradients[i]).collect(Collectors.toList());

                    //phrases
                    List<String> goodPhrases = mixedSplitExtractor.getGoodNgrams(focusSet, blackList, k,
                            gradientsForValidation, numSeeds,topNs[k]);
                    System.out.println("phrases extracted for class " + k + " (" + labelTranslator.toExtLabel(k) + "):");
                    System.out.println(goodPhrases);
                    blackList.addAll(goodPhrases);


                    List<Pair<String, SearchResponse>> searchResponseList = goodPhrases.stream().parallel()
                            .map(phrase -> new Pair<>(phrase, index.matchPhrase(index.getBodyField(),
                                    phrase, trainIdTranslator.getAllExtIds(), 0)))
                            .collect(Collectors.toList());

                    for (Pair<String, SearchResponse> pair : searchResponseList) {
                        String phrase = pair.getFirst();
                        SearchResponse response = pair.getSecond();
                        int featureIndex = featureList.nextAvailable();
                        for (SearchHit hit : response.getHits().getHits()) {
                            String indexId = hit.getId();
                            int algorithmId = trainIdTranslator.toIntId(indexId);
                            float score = hit.getScore();
                            dataSet.setFeatureValue(algorithmId, featureIndex, score);
                        }

                        Ngram ngram = new Ngram();
                        //todo
                        ngram.setField("body");
                        ngram.setSlop(0);
                        ngram.setName(phrase);
                        ngram.setNgram(phrase);
                        ngram.getSettings().put("source","matching_score");
                        featureList.add(ngram);
                    }

                }


                statsWriter.write("iteration = " + iteration);
                statsWriter.write(",");
                statsWriter.write("focus set = " + focusSet.getAll());
                statsWriter.write(",");
                statsWriter.write("number of features = " + featureList.size());
                statsWriter.newLine();


            }

        }

        statsWriter.close();
        File serializedModel =  new File(archive,modelName);
        boosting.serialize(serializedModel);
        System.out.println("model saved to "+serializedModel.getAbsolutePath());
        System.out.println("accuracy on training set = "+ Accuracy.accuracy(boosting,
                dataSet));
        System.out.println("time spent = "+stopWatch);

    }




    static void build(Config config, MultiLabelIndex index) throws Exception{
        int numDocsInIndex = index.getNumDocs();
        String[] trainIndexIds = sampleTrain(config,index);
        System.out.println("number of training documents = "+trainIndexIds.length);
        IdTranslator trainIdTranslator = loadIdTranslator(trainIndexIds);
        FeatureList featureList = new FeatureList();
        LabelTranslator trainLabelTranslator = loadTrainLabelTranslator(index,trainIndexIds);
        if (config.getBoolean("useInitialFeatures")){
            addInitialFeatures(config,index,featureList,trainIndexIds);
        }

        MultiLabelClfDataSet trainDataSet = loadTrainData(config,index,featureList, trainIdTranslator, trainLabelTranslator);

        trainModel(config,trainDataSet,featureList,index, trainIdTranslator);

        //only keep used columns
        List<Integer> columns = IntStream.range(0,featureList.size()).mapToObj(i-> i)
                .collect(Collectors.toList());
        MultiLabelClfDataSet trimmedTrainDataSet = DataSetUtil.trim(trainDataSet,columns);
        trimmedTrainDataSet.setFeatureList(featureList);
        saveDataSet(config, trimmedTrainDataSet, config.getString("archive.trainingSet"));
        if (config.getBoolean("archive.dumpFields")){
            dumpTrainFields(config, index, trainIdTranslator);
        }

        String[] testIndexIds = sampleTest(numDocsInIndex,trainIndexIds);
        IdTranslator testIdTranslator = loadIdTranslator(testIndexIds);
        LabelTranslator testLabelTranslator = loadTestLabelTranslator(index,testIndexIds,trainLabelTranslator);

        MultiLabelClfDataSet testDataSet = loadTestData(config,index,featureList,testIdTranslator,testLabelTranslator);
        testDataSet.setFeatureList(featureList);
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
