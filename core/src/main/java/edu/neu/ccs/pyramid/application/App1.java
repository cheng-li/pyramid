package edu.neu.ccs.pyramid.application;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.google.common.collect.ConcurrentHashMultiset;
import com.google.common.collect.Multiset;
import edu.neu.ccs.pyramid.configuration.Config;
import edu.neu.ccs.pyramid.dataset.*;
import edu.neu.ccs.pyramid.elasticsearch.ESIndex;
import edu.neu.ccs.pyramid.elasticsearch.FeatureLoader;
import edu.neu.ccs.pyramid.feature.*;
import edu.neu.ccs.pyramid.feature_extraction.NgramEnumerator;
import edu.neu.ccs.pyramid.feature_extraction.NgramTemplate;
import edu.neu.ccs.pyramid.feature_extraction.StumpSelector;
import edu.neu.ccs.pyramid.util.BoundedBlockPriorityQueue;
import edu.neu.ccs.pyramid.util.Pair;
import edu.neu.ccs.pyramid.util.Serialization;
import org.apache.commons.io.FileUtils;
import org.elasticsearch.search.aggregations.bucket.terms.Terms;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.*;
import java.util.logging.*;
import java.util.regex.Pattern;
import java.util.stream.Collectors;

/**
 * for multi label dataset,
 * dump feature matrix with initial features and ngram features
 * Created by chengli on 6/12/15.
 */
public class App1 {

    public static void main(String[] args) throws Exception{

        if (args.length !=1){
            throw new IllegalArgumentException("Please specify a properties file.");
        }

        Config config = new Config(args[0]);
        main(config);
    }

    public static void main(Config config) throws Exception{
        Logger logger = Logger.getAnonymousLogger();
        String logFile = config.getString("output.log");
        FileHandler fileHandler = null;
        if (!logFile.isEmpty()){
            new File(logFile).getParentFile().mkdirs();
            //todo should append?
            fileHandler = new FileHandler(logFile, true);
            java.util.logging.Formatter formatter = new SimpleFormatter();
            fileHandler.setFormatter(formatter);
            logger.addHandler(fileHandler);
            logger.setUseParentHandlers(false);
        }
        
        logger.info(config.toString());
        File output = new File(config.getString("output.folder"));
        output.mkdirs();

        if (config.getBoolean("createTrainSet")){
            try (ESIndex index = loadIndex(config, logger)){
                createTrainSet(config, index, logger);
            }

        }


        if (config.getBoolean("createTestSet")){
            try (ESIndex index = loadIndex(config, logger)){
                createTestSet(config, index, logger);
            }

        }

        if (config.getBoolean("createValidSet")){
            try (ESIndex index = loadIndex(config, logger)){
                createValidSet(config, index, logger);
            }

        }

        if (fileHandler!=null){
            fileHandler.close();
        }
    }

    static ESIndex loadIndex(Config config, Logger logger) throws Exception{



        ESIndex.Builder builder = new ESIndex.Builder()
                .setIndexName(config.getString("index.indexName"))
                .setClusterName(config.getString("index.clusterName"))
                .setClientType(config.getString("index.clientType"))

                .setDocumentType(config.getString("index.documentType"));






        if (config.getString("index.clientType").equals("transport")){
            String[] hosts = config.getString("index.hosts").split(Pattern.quote(","));
            String[] ports = config.getString("index.ports").split(Pattern.quote(","));
            builder.addHostsAndPorts(hosts,ports);
        }
        ESIndex index = builder.build();
        logger.info("index loaded");
        logger.info("there are "+index.getNumDocs()+" documents in the index.");
        return index;
    }


    static String[] getDocsForSplitFromQuery(ESIndex index, String query){
        List<String> docs = index.matchStringQuery(query);
        return docs.toArray(new String[docs.size()]);
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
                                   String[] ids, Logger logger) throws Exception{
        String featureFieldPrefix = config.getString("train.feature.featureFieldPrefix");
        Set<String> prefixes = Arrays.stream(featureFieldPrefix.split(",")).map(String::trim).collect(Collectors.toSet());

        Set<String> allFields = index.listAllFields();
        List<String> featureFields = allFields.stream().
                filter(field -> matchPrefixes(field,prefixes)).
                collect(Collectors.toList());
        logger.info("all possible initial features:"+featureFields);

        for (String field: featureFields){
            String featureType = index.getFieldType(field);
            if (featureType.equalsIgnoreCase("text")||featureType.equalsIgnoreCase("keyword")){
                CategoricalFeatureExpander expander = new CategoricalFeatureExpander();
                expander.setStart(featureList.size());
                expander.setVariableName(field);
                expander.putSetting("source","field");

                Collection<org.elasticsearch.search.aggregations.bucket.terms.Terms.Bucket> buckets= index.termAggregation(field,ids);
                Set<String> categories = buckets.stream().map(Terms.Bucket::getKeyAsString).collect(Collectors.toSet());

                for (String category: categories){
                    expander.addCategory(category);
                }
//                for (String id: ids){
//                    String category = index.getStringField(id, field);
//                    expander.addCategory(category);
//                }
                List<CategoricalFeature> group = expander.expand();
                boolean toAdd = true;
                if (config.getBoolean("train.feature.categFeature.filter")){
                    double threshold = config.getDouble("train.feature.categFeature.percentThreshold");
                    int numCategories = group.size();
                    if (numCategories> ids.length*threshold){
                        toAdd=false;
                        logger.info("field "+field+" has too many categories "
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
                feature.getSettings().put("source", "field");
                featureList.add(feature);
            }
        }

    }

    static boolean interesting(Multiset<Ngram> allNgrams, Ngram candidate, int count){
        for (int slop = 0;slop<candidate.getSlop();slop++){
            Ngram toCheck = new Ngram();
            toCheck.setInOrder(candidate.isInOrder());
            toCheck.setField(candidate.getField());
            toCheck.setNgram(candidate.getNgram());
            toCheck.setSlop(slop);
            toCheck.setName(candidate.getName());
            if (allNgrams.count(toCheck)==count){
                return false;
            }
        }
        return true;
    }

    static Set<Ngram> gather(Config config, ESIndex index,
                             String[] ids, Logger logger) throws Exception{

        File metaDataFolder = new File(config.getString("output.folder"),"meta_data");
        metaDataFolder.mkdirs();

        Multiset<Ngram> allNgrams = ConcurrentHashMultiset.create();
        List<Integer> ns = config.getIntegers("train.feature.ngram.n");
        double minDf = config.getDouble("train.feature.ngram.minDf");
        int minDFrequency = (int)Math.floor(ids.length*minDf);
        List<String> fields = config.getStrings("train.feature.ngram.extractionFields");
        List<Integer> slops = config.getIntegers("train.feature.ngram.slop");
        boolean inorder = config.getBoolean("train.feature.ngram.inOrder");
        boolean allowDuplicates = config.getBoolean("train.feature.ngram.allowDuplicateWords");
        for (String field: fields){
            for (int n: ns){
                for (int slop:slops){
                    logger.info("gathering "+n+ "-grams from field "+field+" with slop "+slop+" and minDf "+minDf+ ", (actual frequency threshold = "+minDFrequency+")");
                    NgramTemplate template = new NgramTemplate(field,n,slop);
                    Multiset<Ngram> ngrams = NgramEnumerator.gatherNgram(index, ids, template, minDFrequency);
                    logger.info("gathered "+ngrams.elementSet().size()+ " ngrams");
                    int newCounter = 0;
                    for (Multiset.Entry<Ngram> entry: ngrams.entrySet()){
                        Ngram ngram = entry.getElement();
                        ngram.setInOrder(inorder);
                        int count = entry.getCount();
                        if (interesting(allNgrams,ngram,count)){
                            if (allowDuplicates) {
                                allNgrams.add(ngram, count);
                                newCounter += 1;
                            }else{
                                if (!ngram.hasDuplicate()){
                                    allNgrams.add(ngram, count);
                                    newCounter += 1;
                                }
                            }
                        }

                    }
                    logger.info(newCounter+" are really new");
                }
            }
        }
        logger.info("there are "+allNgrams.elementSet().size()+" ngrams in total");
//        BufferedWriter bufferedWriter = new BufferedWriter(new FileWriter(new File(metaDataFolder,"all_ngrams.txt")));
//        for (Multiset.Entry<Ngram> ngramEntry: allNgrams.entrySet()){
//            bufferedWriter.write(ngramEntry.getElement().toString());
//            bufferedWriter.write("\t");
//            bufferedWriter.write(""+ngramEntry.getCount());
//            bufferedWriter.newLine();
//        }
//
//        bufferedWriter.close();
//
//        //for serialization
//        Set<Ngram> uniques = new HashSet<>();
//        uniques.addAll(allNgrams.elementSet());
//        Serialization.serialize(uniques, new File(metaDataFolder, "all_ngrams.ser"));
        return allNgrams.elementSet();
    }

    private static List<Ngram> addNgramFromFile(Config config, ESIndex index, Logger logger) throws IOException {
        List<Ngram> ngrams = new ArrayList<>();
        String externalNgramFile = config.getString("train.feature.externalNgramFile");
        List<String> lines = FileUtils.readLines(new File(externalNgramFile));
        List<String> fields = config.getStrings("train.feature.ngram.extractionFields");
        String analyzer = config.getString("train.feature.analyzer");
        for (String field: fields){
            for (String line: lines){
                Ngram ngram = index.analyze(line,analyzer);
                ngram.setField(field);
                ngrams.add(ngram);
            }
        }
        logger.info("ngrams collected from file "+externalNgramFile);
        logger.info(ngrams.toString());
        return ngrams;
    }

    private static void addCodeDescription(Config config, ESIndex index, FeatureList featureList) throws Exception{
        String file = config.getString("train.feature.codeDesc.File");
        List<String> lines = FileUtils.readLines(new File(file));
        String analyzer = config.getString("train.feature.codeDesc.analyzer");
        String field = config.getString("train.feature.codeDesc.matchField");
        int percentage = config.getInt("train.feature.codeDesc.minMatchPercentage");
        for (String line: lines){
            List<String> terms = index.analyzeString(line, analyzer);
            CodeDescription codeDescription = new CodeDescription(terms, percentage, field);
            featureList.add(codeDescription);
        }
    }


    static void addNgramFeatures(FeatureList featureList, Set<Ngram> ngrams){
        List<Ngram> list = new ArrayList<>(ngrams);
        Collections.sort(list);
        list.stream().forEach(ngram -> {
            ngram.getSettings().put("source","matching_score");
            featureList.add(ngram);
        });
    }

    //todo keep track of feature types(numerical /binary)
    static MultiLabelClfDataSet loadData(Config config, ESIndex index,
                                         FeatureList featureList,
                                         IdTranslator idTranslator, int totalDim,
                                         LabelTranslator labelTranslator,
                                         String docFilter) throws Exception{

        File metaDataFolder = new File(config.getString("output.folder"),"meta_data");
        Config savedConfig = new Config(new File(metaDataFolder, "saved_config_app1"));

        int numDataPoints = idTranslator.numData();
        int numClasses = labelTranslator.getNumClasses();
        MultiLabelClfDataSet dataSet = MLClfDataSetBuilder.getBuilder()
                .numDataPoints(numDataPoints).numFeatures(totalDim)
                .numClasses(numClasses)
                .density(Density.SPARSE_RANDOM)
                .missingValue(savedConfig.getBoolean("train.feature.missingValue")).build();

        for (int l=0;l<labelTranslator.getNumClasses();l++){
            String label = labelTranslator.toExtLabel(l);
            String labelField = config.getString("train.label.field");
            List<String> ids = index.termFilter(labelField,label,docFilter, idTranslator.numData());
            for (String id: ids){
                int intId = idTranslator.toIntId(id);
                dataSet.addLabel(intId, l);
            }
        }


        String matchScoreTypeString = savedConfig.getString("train.feature.ngram.matchScoreType");

        FeatureLoader.MatchScoreType matchScoreType;
        Map<String, float[]> fieldLength=null;

        switch (matchScoreTypeString){
            case "es_original":
                matchScoreType= FeatureLoader.MatchScoreType.ES_ORIGINAL;
                break;
            case "binary":
                matchScoreType= FeatureLoader.MatchScoreType.BINARY;
                break;
            case "frequency":
                matchScoreType= FeatureLoader.MatchScoreType.FREQUENCY;
                break;
            case "tfifl":
                matchScoreType= FeatureLoader.MatchScoreType.TFIFL;
                fieldLength = new HashMap<>();
                for (String field: config.getStrings("train.feature.ngram.extractionFields")){

                    float[] arr = new float[idTranslator.numData()];
                    for (int i=0;i<idTranslator.numData();i++){
                        String extId = idTranslator.toExtId(i);
                        arr[i]=index.getFloatField(extId, field+"_field_length");
                    }
                    fieldLength.put(field, arr);
                }


                break;
            default:
                throw new IllegalArgumentException("unknown ngramMatchScoreType");
        }

        FeatureLoader.loadFeatures(index, dataSet, featureList, idTranslator, matchScoreType, docFilter, fieldLength);

        dataSet.setIdTranslator(idTranslator);
        dataSet.setLabelTranslator(labelTranslator);
        return dataSet;
    }


    static LabelTranslator loadTrainLabelTranslator(Config config, ESIndex index, String[] trainIndexIds,
                                                    Logger logger) throws Exception{
        Collection<Terms.Bucket> buckets = index.termAggregation(config.getString("train.label.field"), trainIndexIds);
        if (config.getBoolean("train.label.filterByPrefix")){
            String prefix = config.getString("train.label.filter.prefix");
            buckets = buckets.stream().filter(bucket -> bucket.getKeyAsString().startsWith(prefix)).collect(Collectors.toList());
        }
        logger.info("there are "+buckets.size()+" classes in the training set.");
        List<String> labels = new ArrayList<>();
        logger.info("label distribution in training set:");
        StringBuilder stringBuilder = new StringBuilder();
        int trainLabelMinDF = config.getInt("train.label.minDF");
        for (Terms.Bucket bucket: buckets){
            if (bucket.getDocCount() >= trainLabelMinDF){
                stringBuilder.append(bucket.getKey());
                stringBuilder.append(":");
                stringBuilder.append(bucket.getDocCount());
                stringBuilder.append(", ");
                labels.add(bucket.getKeyAsString());
            }
        }
        logger.info(stringBuilder.toString());

        String labelOrder = config.getString("train.label.order");
        if (labelOrder.equals("alphabetical")){
            Collections.sort(labels);
        }

        LabelTranslator labelTranslator = new LabelTranslator(labels);
//        logger.info(labelTranslator);
        return labelTranslator;
    }

    static LabelTranslator loadTestLabelTranslator(Config config, ESIndex index, String[] testIndexIds,
                                                   LabelTranslator trainLabelTranslator, Logger logger){

        File metaDataFolder = new File(config.getString("output.folder"),"meta_data");
        Config savedConfig = new Config(new File(metaDataFolder, "saved_config_app1"));
        boolean testConsiderNew = config.getBoolean("test.considerNewLabel");

        List<String> extLabels = new ArrayList<>();
        for (int i=0;i<trainLabelTranslator.getNumClasses();i++){
            extLabels.add(trainLabelTranslator.toExtLabel(i));
        }

        Collection<Terms.Bucket> buckets = index.termAggregation(savedConfig.getString("train.label.field"), testIndexIds);
        if (savedConfig.getBoolean("train.label.filterByPrefix")){
            String prefix = savedConfig.getString("train.label.filter.prefix");
            buckets = buckets.stream().filter(bucket -> bucket.getKeyAsString().startsWith(prefix)).collect(Collectors.toList());
        }

        logger.info("label distribution in data set:");
        StringBuilder stringBuilder = new StringBuilder();
        if (testConsiderNew){
            List<String> newLabels = new ArrayList<>();
            for (Terms.Bucket bucket: buckets){
                stringBuilder.append(bucket.getKey());
                stringBuilder.append(":");
                stringBuilder.append(bucket.getDocCount());
                stringBuilder.append(", ");
                if (!extLabels.contains(bucket.getKey())){
                    extLabels.add(bucket.getKeyAsString());
                    newLabels.add(bucket.getKeyAsString());
                }
            }
            if (!newLabels.isEmpty()){
                logger.warning("found new labels in data set: "+newLabels);
            }
        }else{
            for (Terms.Bucket bucket: buckets){
                stringBuilder.append(bucket.getKey());
                stringBuilder.append(":");
                stringBuilder.append(bucket.getDocCount());
                stringBuilder.append(", ");
            }

        }

        logger.info(stringBuilder.toString());

        return new LabelTranslator(extLabels);
    }

    static LabelTranslator loadValidLabelTranslator(Config config, ESIndex index, String[] validIndexIds,
                                                   LabelTranslator trainLabelTranslator, Logger logger){

        File metaDataFolder = new File(config.getString("output.folder"),"meta_data");
        Config savedConfig = new Config(new File(metaDataFolder, "saved_config_app1"));
        boolean validConsiderNew = config.getBoolean("valid.considerNewLabel");

        List<String> extLabels = new ArrayList<>();
        for (int i=0;i<trainLabelTranslator.getNumClasses();i++){
            extLabels.add(trainLabelTranslator.toExtLabel(i));
        }

        Collection<Terms.Bucket> buckets = index.termAggregation(savedConfig.getString("train.label.field"), validIndexIds);
        if (savedConfig.getBoolean("train.label.filterByPrefix")){
            String prefix = savedConfig.getString("train.label.filter.prefix");
            buckets = buckets.stream().filter(bucket -> bucket.getKeyAsString().startsWith(prefix)).collect(Collectors.toList());
        }

        logger.info("label distribution in data set:");
        StringBuilder stringBuilder = new StringBuilder();
        if (validConsiderNew){
            List<String> newLabels = new ArrayList<>();
            for (Terms.Bucket bucket: buckets){
                stringBuilder.append(bucket.getKey());
                stringBuilder.append(":");
                stringBuilder.append(bucket.getDocCount());
                stringBuilder.append(", ");
                if (!extLabels.contains(bucket.getKey())){
                    extLabels.add(bucket.getKeyAsString());
                    newLabels.add(bucket.getKeyAsString());
                }
            }
            if (!newLabels.isEmpty()){
                logger.warning("found new labels in data set: "+newLabels);
            }
        }else{
            for (Terms.Bucket bucket: buckets){
                stringBuilder.append(bucket.getKey());
                stringBuilder.append(":");
                stringBuilder.append(bucket.getDocCount());
                stringBuilder.append(", ");
            }

        }

        logger.info(stringBuilder.toString());

        return new LabelTranslator(extLabels);
    }











//    static void getNgramDistributions(Config config, ESIndex index, String[] ids,LabelTranslator labelTranslator ) throws Exception{
//        File metaDataFolder = new File(config.getString("output.folder"),"meta_data");
//        metaDataFolder.mkdirs();
//
//        logger.info("generating ngram distributions");
//        StopWatch stopWatch = new StopWatch();
//        stopWatch.start();
//        File file = new File(metaDataFolder,"all_ngrams.ser");
//        Set<Ngram> ngrams= (Set) Serialization.deserialize(file);
//        String labelField = config.getString("train.label.labelField");
//        long[] labelDistribution = LabelDistribution.getLabelDistribution(index,labelField,ids,labelTranslator);
//        List<FeatureDistribution> distributions = ngrams.stream().parallel()
//                .map(ngram -> new FeatureDistribution(ngram, index, labelField, ids, labelTranslator,labelDistribution))
//                .collect(Collectors.toList());
//        Serialization.serialize(distributions,new File(metaDataFolder,"distributions.ser"));
//        BufferedWriter bufferedWriter = new BufferedWriter(new FileWriter(new File(metaDataFolder,"distributions.txt")));
//        for (FeatureDistribution distribution: distributions){
//            bufferedWriter.write(distribution.toString());
//            bufferedWriter.newLine();
//        }
//
//        bufferedWriter.close();
//        logger.info("done");
//        logger.info("time spent on generating distributions = "+stopWatch);
//    }
    
    static void generateMetaData(Config config, ESIndex index, Logger logger) throws Exception{
        logger.info("generating meta data");
        File metaDataFolder = new File(config.getString("output.folder"),"meta_data");
        metaDataFolder.mkdirs();

        config.store(new File(metaDataFolder, "saved_config_app1"));

        String[] trainIndexIds;

        trainIndexIds = getDocsForSplitFromQuery(index, config.getString("train.splitQuery"));



        LabelTranslator trainLabelTranslator = loadTrainLabelTranslator(config, index, trainIndexIds, logger);
        Serialization.serialize(trainLabelTranslator,new File(metaDataFolder,"label_translator.ser"));
        FileUtils.writeStringToFile(new File(metaDataFolder,"label_translator.txt"),trainLabelTranslator.toString());

        FeatureList featureList = new FeatureList();
        if (config.getBoolean("train.feature.useInitialFeatures")){
            addInitialFeatures(config,index,featureList,trainIndexIds, logger);
        }

        if (config.getBoolean("train.feature.useCodeDescription")){
            addCodeDescription(config, index, featureList);
        }


        Set<Ngram> ngrams = new HashSet<>();
        ngrams.addAll(gather(config,index,trainIndexIds, logger));

        if (config.getBoolean("train.feature.filterNgramsByKeyWords")){
            ngrams = keywordsFilter(config,index,ngrams);
        }

        if (config.getBoolean("train.feature.filterNgramsByRegex")){
            ngrams = regexFilter(config,ngrams);
        }

        if (config.getBoolean("train.feature.addExternalNgrams")){
            ngrams.addAll(addNgramFromFile(config, index, logger));
        }
//        if (config.getBoolean("train.feature.generateDistribution")){
//            getNgramDistributions(config,index,trainIndexIds,trainLabelTranslator);
//        }

        addNgramFeatures(featureList,ngrams);

        Serialization.serialize(featureList,new File(metaDataFolder,"feature_list.ser"));
        try (BufferedWriter bufferedWriter = new BufferedWriter(new FileWriter(new File(metaDataFolder,"feature_list.txt")))
        ){
            for (Feature feature: featureList.getAll()){
                bufferedWriter.write(feature.toString());
                bufferedWriter.newLine();
            }
        }


        if (config.getBoolean("train.feature.ngram.selection")){
            ngramSelection(config, index, config.getString("train.splitQuery"), logger);
        }





        logger.info("meta data generated");


    }

    static void createDataSet(Config config, ESIndex index, String[] indexIds, String datasetName,
                              String docFilter, Logger logger, String trainTestValid) throws Exception{
//        String splitValueAll = splitListToString(splitValues);


        logger.info("creating data set "+datasetName);
        File metaDataFolder = new File(config.getString("output.folder"),"meta_data");
        IdTranslator idTranslator = loadIdTranslator(indexIds);
        String archive = config.getString("output.folder");
        LabelTranslator trainLabelTranslator = (LabelTranslator)Serialization.deserialize(new File(metaDataFolder,"label_translator.ser"));
        FeatureList featureList = (FeatureList)Serialization.deserialize(new File(metaDataFolder,"feature_list.ser"));
        MultiLabelClfDataSet dataSet=null;
        if (trainTestValid.equals("test")){
            LabelTranslator labelTranslator = loadTestLabelTranslator(config, index, indexIds, trainLabelTranslator, logger);
            dataSet = loadData(config, index, featureList, idTranslator, featureList.size(), labelTranslator, docFilter);
        }
        if (trainTestValid.equals("train")){
                dataSet = loadData(config, index, featureList, idTranslator, featureList.size(), trainLabelTranslator, docFilter);
        }
        if (trainTestValid.equals("valid")){
                LabelTranslator labelTranslator = loadValidLabelTranslator(config, index, indexIds, trainLabelTranslator,logger);
                dataSet = loadData(config, index, featureList, idTranslator,featureList.size(), labelTranslator, docFilter);
        }
        dataSet.setFeatureList(featureList);
        File dataFile = new File(new File(archive,"data_sets"),datasetName);

        TRECFormat.save(dataSet,dataFile);
        logger.info("data set "+datasetName+" created");

        ObjectMapper objectMapper = new ObjectMapper();
        objectMapper.writeValue(new File(dataFile,"data_config.json"),config);


    }

    static void createTrainSet(Config config, ESIndex index, Logger logger) throws Exception{
        generateMetaData(config, index, logger);
        String[] indexIds = getDocsForSplitFromQuery(index, config.getString("train.splitQuery"));

        createDataSet(config, index, indexIds,config.getString("output.trainFolder"),
                config.getString("train.splitQuery"), logger, "train");
    }

    static void createTestSet(Config config, ESIndex index, Logger logger) throws Exception{
        String[] indexIds = getDocsForSplitFromQuery(index, config.getString("test.splitQuery"));
        createDataSet(config, index, indexIds,config.getString("output.testFolder"),
                config.getString("test.splitQuery"), logger,"test");
    }

    static void createValidSet(Config config, ESIndex index, Logger logger) throws Exception{
        String[] indexIds = getDocsForSplitFromQuery(index, config.getString("valid.splitQuery"));
        createDataSet(config, index, indexIds, config.getString("output.validFolder"),
                config.getString("valid.splitQuery"), logger, "valid");
    }

//    public static String splitListToString(List<String> splitValues){
//        String splitValueAll = "";
//        for (int i=0;i<splitValues.size();i++){
//            splitValueAll = splitValueAll+splitValues.get(i);
//            if (i<splitValues.size()-1){
//                splitValueAll = splitValueAll+"_";
//            }
//        }
//        return splitValueAll;
//    }

    /**
     * filter ngrams by given unigrams in the file
     * do not filter unigram candidates
     */
    private static Set<Ngram> keywordsFilter(Config config, ESIndex index, Set<Ngram> ngrams) throws IOException {
        String externalKeywordsFile = config.getString("train.feature.filterNgrams.keyWordsFile");
        List<String> lines = FileUtils.readLines(new File(externalKeywordsFile));
        String analyzer = config.getString("train.feature.analyzer");
        Set<String> keywords = new HashSet<>();
        for (String line: lines){
            keywords.add(index.analyze(line, analyzer).getNgram());
        }

        return ngrams.stream().parallel().filter(ngram-> ngram.getN()==1||containsKeyWords(ngram,keywords))
                .collect(Collectors.toSet());
    }

    private static boolean containsKeyWords(Ngram ngram, Set<String> keywords){
        String[] terms = ngram.getTerms();
        for (String term: terms){
            if (keywords.contains(term)){
                return true;
            }
        }
        return false;
    }


    /**
     *
     * @param config
     * @param ngrams
     * @return ngrams that do not match the regular expression
     */
    private static Set<Ngram> regexFilter(Config config, Set<Ngram> ngrams){
        String regex = config.getString("train.feature.filterNgrams.regex");
        return ngrams.parallelStream().filter(ngram->!ngram.getNgram().matches(regex)).collect(Collectors.toSet());
    }


    /**
     *
     * @return into 2d arrary: num label * num data
     */
    private static double[][] loadLabels(Config config, ESIndex index,
                                         IdTranslator idTranslator,
                                         LabelTranslator labelTranslator,
                                         String docFilter) throws Exception{
        File metaDataFolder = new File(config.getString("output.folder"),"meta_data");
        Config savedConfig = new Config(new File(metaDataFolder, "saved_config_app1"));

        int numDataPoints = idTranslator.numData();
        int numClasses = labelTranslator.getNumClasses();
        double[][] labels = new double[numClasses][numDataPoints];
        for (int l=0;l<labelTranslator.getNumClasses();l++){
            String label = labelTranslator.toExtLabel(l);
            String labelField = config.getString("train.label.field");
            List<String> ids = index.termFilter(labelField,label,docFilter, idTranslator.numData());
            for (String id: ids){
                int intId = idTranslator.toIntId(id);
                labels[l][intId] = 1;
            }
        }

        return labels;
    }

    private static void ngramSelection(Config config, ESIndex index,
                                       String docFilter,
                                       Logger logger)throws Exception{

        logger.info("start ngram selection");
        File metaDataFolder = new File(config.getString("output.folder"),"meta_data");
        FeatureLoader.MatchScoreType matchScoreType;
        String matchScoreTypeString = config.getString("train.feature.ngram.matchScoreType");
        String[] indexIds = getDocsForSplitFromQuery(index, config.getString("train.splitQuery"));
        IdTranslator idTranslator = loadIdTranslator(indexIds);
        LabelTranslator labelTranslator = (LabelTranslator)Serialization.deserialize(new File(metaDataFolder,"label_translator.ser"));

        final Map<String, float[]> fieldLength=new HashMap<>();

        switch (matchScoreTypeString){
            case "es_original":
                matchScoreType= FeatureLoader.MatchScoreType.ES_ORIGINAL;
                break;
            case "binary":
                matchScoreType= FeatureLoader.MatchScoreType.BINARY;
                break;
            case "frequency":
                matchScoreType= FeatureLoader.MatchScoreType.FREQUENCY;
                break;
            case "tfifl":
                matchScoreType= FeatureLoader.MatchScoreType.TFIFL;

                for (String field: config.getStrings("train.feature.ngram.extractionFields")){

                    float[] arr = new float[idTranslator.numData()];
                    for (int i=0;i<idTranslator.numData();i++){
                        String extId = idTranslator.toExtId(i);
                        arr[i]=index.getFloatField(extId, field+"_field_length");
                    }
                    fieldLength.put(field, arr);
                }


                break;
            default:
                throw new IllegalArgumentException("unknown ngramMatchScoreType");
        }

        double[][] labels = loadLabels(config, index, idTranslator, labelTranslator, docFilter);
        int numLabels = labels.length;
        int toKeep = config.getInt("train.feature.ngram.selectPerLabel");
        List<BoundedBlockPriorityQueue<Pair<Ngram, Double>>> queues = new ArrayList<>();
        Comparator<Pair<Ngram, Double>> comparator = Comparator.comparing(p->p.getSecond());
        for (int l=0;l<numLabels;l++){
            queues.add(new BoundedBlockPriorityQueue<>(toKeep, comparator));
        }

        FeatureList featureList = (FeatureList)Serialization.deserialize(new File(metaDataFolder,"feature_list.ser"));

        featureList.getAll().stream().parallel()
                .filter(feature->feature instanceof Ngram).map(feature->(Ngram)feature)
                .filter(ngram -> ngram.getN()>1 )
                .forEach(ngram ->{
                    double[] scores = StumpSelector.scores(index, labels, ngram, idTranslator, matchScoreType, docFilter, fieldLength);
                    for (int l=0;l<numLabels;l++){
                        queues.get(l).add(new Pair<>(ngram, scores[l]));
                    }
                });

        Set<Ngram> kept = new HashSet<>();
        StringBuilder stringBuilder = new StringBuilder();
        for (int l=0;l<numLabels;l++){
            stringBuilder.append("-------------------------").append("\n");
            stringBuilder.append(labelTranslator.toExtLabel(l)).append(":").append("\n");
            BoundedBlockPriorityQueue<Pair<Ngram, Double>> queue = queues.get(l);
            while(queue.size()>0){
                Ngram ngram = queue.poll().getFirst();
                kept.add(ngram);
                stringBuilder.append(ngram.getNgram()).append(", ");
            }
            stringBuilder.append("\n");
        }

        File selectionFile = new File(metaDataFolder,"selected_ngrams.txt");

        FileUtils.writeStringToFile(selectionFile, stringBuilder.toString());

        logger.info("finish ngram selection");
        logger.info("selected ngrams are written to "+selectionFile.getAbsolutePath());

        // after feature selection, overwrite the feature_list.ser file; rename old files

        FeatureList selectedFeatures = new FeatureList();
        for (Feature feature: featureList.getAll()){
            if (!(feature instanceof Ngram)){
                selectedFeatures.add(feature);
            }
            if ((feature instanceof Ngram) && ((Ngram)feature).getN()==1){
                selectedFeatures.add(feature);
            }

            if ((feature instanceof Ngram) && ((Ngram)feature).getN()>1 && kept.contains(feature)){
                selectedFeatures.add(feature);
            }
        }

        FileUtils.copyFile(new File(metaDataFolder,"feature_list.ser"), new File(metaDataFolder,"feature_list_all.ser"));
        FileUtils.copyFile(new File(metaDataFolder,"feature_list.txt"), new File(metaDataFolder,"feature_list_all.txt"));

        Serialization.serialize(selectedFeatures,new File(metaDataFolder,"feature_list.ser"));
        try (BufferedWriter bufferedWriter = new BufferedWriter(new FileWriter(new File(metaDataFolder,"feature_list.txt")))
        ){
            for (Feature feature: selectedFeatures.getAll()){
                bufferedWriter.write(feature.toString());
                bufferedWriter.newLine();
            }
        }

    }


}
