package edu.neu.ccs.pyramid.application;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.google.common.collect.ConcurrentHashMultiset;
import com.google.common.collect.Multiset;
import edu.neu.ccs.pyramid.configuration.Config;
import edu.neu.ccs.pyramid.dataset.*;
import edu.neu.ccs.pyramid.elasticsearch.ESIndex;
import edu.neu.ccs.pyramid.elasticsearch.FeatureLoader;
import edu.neu.ccs.pyramid.elasticsearch.MultiLabelIndex;
import edu.neu.ccs.pyramid.feature.*;
import edu.neu.ccs.pyramid.feature_extraction.NgramEnumerator;
import edu.neu.ccs.pyramid.feature_extraction.NgramTemplate;
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
    private static final Logger logger = Logger.getLogger(App1.class.getName());
    
    public static void main(String[] args) throws Exception{

        if (args.length !=1){
            throw new IllegalArgumentException("Please specify a properties file.");
        }

        Config config = new Config(args[0]);
        main(config);
    }

    public static void main(Config config) throws Exception{
        String logFile = config.getString("out.log");
        if (!logFile.isEmpty()){
            new File(logFile).getParentFile().mkdirs();
            //todo should append?
            FileHandler fileHandler = new FileHandler(logFile, true);
            java.util.logging.Formatter formatter = new SimpleFormatter();
            fileHandler.setFormatter(formatter);
            logger.addHandler(fileHandler);
        }
        
        logger.info(config.toString());
        File output = new File(config.getString("output.folder"));
        output.mkdirs();

        if (config.getBoolean("createTrainSet")){
            try (MultiLabelIndex index = loadIndex(config)){
                createTrainSet(config, index);
            }

        }


        if (config.getBoolean("createTestSet")){
            try (MultiLabelIndex index = loadIndex(config)){
                createTestSet(config, index);
            }

        }


    }

    static MultiLabelIndex loadIndex(Config config) throws Exception{
        MultiLabelIndex.Builder builder = new MultiLabelIndex.Builder()
                .setIndexName(config.getString("index.indexName"))
                .setClusterName(config.getString("index.clusterName"))
                .setClientType(config.getString("index.clientType"))
                .setExtMultiLabelField(config.getString("index.labelField"))
                .setDocumentType(config.getString("index.documentType"));
        if (config.getString("index.clientType").equals("transport")){
            String[] hosts = config.getString("index.hosts").split(Pattern.quote(","));
            String[] ports = config.getString("index.ports").split(Pattern.quote(","));
            builder.addHostsAndPorts(hosts,ports);
        }
        MultiLabelIndex index = builder.build();
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
                                   String[] ids) throws Exception{
        String featureFieldPrefix = config.getString("index.featureFieldPrefix");
        Set<String> prefixes = Arrays.stream(featureFieldPrefix.split(",")).map(String::trim).collect(Collectors.toSet());

        Set<String> allFields = index.listAllFields();
        List<String> featureFields = allFields.stream().
                filter(field -> matchPrefixes(field,prefixes)).
                collect(Collectors.toList());
        logger.info("all possible initial features:"+featureFields);

        for (String field: featureFields){
            String featureType = index.getFieldType(field);
            if (featureType.equalsIgnoreCase("string")){
                CategoricalFeatureExpander expander = new CategoricalFeatureExpander();
                expander.setStart(featureList.size());
                expander.setVariableName(field);
                expander.putSetting("source","field");

                Collection<org.elasticsearch.search.aggregations.bucket.terms.Terms.Bucket> buckets= index.termAggregation(field,ids);
                Set<String> categories = buckets.stream().map(Terms.Bucket::getKey).collect(Collectors.toSet());

                for (String category: categories){
                    expander.addCategory(category);
                }
//                for (String id: ids){
//                    String category = index.getStringField(id, field);
//                    expander.addCategory(category);
//                }
                List<CategoricalFeature> group = expander.expand();
                boolean toAdd = true;
                if (config.getBoolean("feature.categFeature.filter")){
                    double threshold = config.getDouble("feature.categFeature.percentThreshold");
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
                             String[] ids) throws Exception{

        File metaDataFolder = new File(config.getString("output.folder"),"meta_data");
        metaDataFolder.mkdirs();

        Multiset<Ngram> allNgrams = ConcurrentHashMultiset.create();
        List<Integer> ns = config.getIntegers("feature.ngram.n");
        int minDf = config.getInt("feature.ngram.minDf");
        List<String> fields = config.getStrings("index.ngramExtractionFields");
        List<Integer> slops = config.getIntegers("feature.ngram.slop");
        for (String field: fields){
            for (int n: ns){
                for (int slop:slops){
                    logger.info("gathering "+n+ "-grams from field "+field+" with slop "+slop+" and minDf "+minDf);
                    NgramTemplate template = new NgramTemplate(field,n,slop);
                    Multiset<Ngram> ngrams = NgramEnumerator.gatherNgram(index, ids, template, minDf);
                    logger.info("gathered "+ngrams.elementSet().size()+ " ngrams");
                    int newCounter = 0;
                    for (Multiset.Entry<Ngram> entry: ngrams.entrySet()){
                        Ngram ngram = entry.getElement();
                        int count = entry.getCount();
                        if (interesting(allNgrams,ngram,count)){
                            allNgrams.add(ngram,count);
                            newCounter += 1;
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

    private static List<Ngram> addNgramFromFile(Config config, ESIndex index) throws IOException {
        List<Ngram> ngrams = new ArrayList<>();
        String externalNgramFile = config.getString("feature.externalNgramFile");
        List<String> lines = FileUtils.readLines(new File(externalNgramFile));
        List<String> fields = config.getStrings("index.ngramExtractionFields");
        String analyzer = config.getString("feature.analyzer");
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
        String file = config.getString("feature.codeDesc.File");
        List<String> lines = FileUtils.readLines(new File(file));
        String analyzer = config.getString("feature.codeDesc.analyzer");
        String field = config.getString("feature.codeDesc.matchField");
        int percentage = config.getInt("feature.codeDesc.minMatchPercentage");
        for (String line: lines){
            List<String> terms = index.analyzeString(line, analyzer);
            CodeDescription codeDescription = new CodeDescription(terms, percentage, field);
            featureList.add(codeDescription);
        }
    }


    static void addNgramFeatures(FeatureList featureList, Set<Ngram> ngrams){
        ngrams.stream().forEach(ngram -> {
            ngram.getSettings().put("source","matching_score");
            featureList.add(ngram);
        });
    }

    //todo keep track of feature types(numerical /binary)
    static MultiLabelClfDataSet loadData(Config config, MultiLabelIndex index,
                                         FeatureList featureList,
                                         IdTranslator idTranslator, int totalDim,
                                         LabelTranslator labelTranslator,
                                         String docFilter) throws Exception{
        int numDataPoints = idTranslator.numData();
        int numClasses = labelTranslator.getNumClasses();
        MultiLabelClfDataSet dataSet = MLClfDataSetBuilder.getBuilder()
                .numDataPoints(numDataPoints).numFeatures(totalDim)
                .numClasses(numClasses).dense(false)
                .missingValue(config.getBoolean("feature.missingValue")).build();
        for(int i=0;i<numDataPoints;i++){
            String dataIndexId = idTranslator.toExtId(i);
            List<String> extMultiLabel = index.getExtMultiLabel(dataIndexId);
            if (config.getBoolean("index.labelFilter")){
                String prefix = config.getString("index.labelFilter.prefix");
                extMultiLabel = extMultiLabel.stream().filter(extLabel -> extLabel.startsWith(prefix)).collect(Collectors.toList());
            }
            for (String extLabel: extMultiLabel){
                int intLabel = labelTranslator.toIntLabel(extLabel);
                dataSet.addLabel(i,intLabel);
            }
        }

        String matchScoreTypeString = config.getString("index.ngramMatchScoreType");

        FeatureLoader.MatchScoreType matchScoreType;

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
                break;
            default:
                throw new IllegalArgumentException("unknown ngramMatchScoreType");
        }

        FeatureLoader.loadFeatures(index, dataSet, featureList, idTranslator, matchScoreType, docFilter);

        dataSet.setIdTranslator(idTranslator);
        dataSet.setLabelTranslator(labelTranslator);
        return dataSet;
    }


    static LabelTranslator loadTrainLabelTranslator(Config config, MultiLabelIndex index, String[] trainIndexIds) throws Exception{
        Collection<Terms.Bucket> buckets = index.termAggregation(config.getString("index.labelField"), trainIndexIds);
        if (config.getBoolean("index.labelFilter")){
            String prefix = config.getString("index.labelFilter.prefix");
            buckets = buckets.stream().filter(bucket -> bucket.getKey().startsWith(prefix)).collect(Collectors.toList());
        }
        logger.info("there are "+buckets.size()+" classes in the training set.");
        List<String> labels = new ArrayList<>();
        logger.info("label distribution in training set:");
        for (Terms.Bucket bucket: buckets){
            System.out.print(bucket.getKey());
            System.out.print(":");
            System.out.print(bucket.getDocCount());
            System.out.print(", ");
            labels.add(bucket.getKey());
        }

        LabelTranslator labelTranslator = new LabelTranslator(labels);
//        logger.info(labelTranslator);
        return labelTranslator;
    }

    static LabelTranslator loadAugmentedLabelTranslator(Config config, MultiLabelIndex index, String[] testIndexIds, LabelTranslator trainLabelTranslator){
        List<String> extLabels = new ArrayList<>();
        for (int i=0;i<trainLabelTranslator.getNumClasses();i++){
            extLabels.add(trainLabelTranslator.toExtLabel(i));
        }

        Collection<Terms.Bucket> buckets = index.termAggregation(config.getString("index.labelField"), testIndexIds);
        if (config.getBoolean("index.labelFilter")){
            String prefix = config.getString("index.labelFilter.prefix");
            buckets = buckets.stream().filter(bucket -> bucket.getKey().startsWith(prefix)).collect(Collectors.toList());
        }
        List<String> newLabels = new ArrayList<>();
        logger.info("label distribution in data set:");
        for (Terms.Bucket bucket: buckets){
            System.out.print(bucket.getKey());
            System.out.print(":");
            System.out.print(bucket.getDocCount());
            System.out.print(", ");
            if (!extLabels.contains(bucket.getKey())){
                extLabels.add(bucket.getKey());
                newLabels.add(bucket.getKey());
            }
        }
        if (!newLabels.isEmpty()){
            logger.warning("found new labels in data set: "+newLabels);
        }
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
//        String labelField = config.getString("index.labelField");
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
    
    static void generateMetaData(Config config, MultiLabelIndex index) throws Exception{
        logger.info("generating meta data");
        File metaDataFolder = new File(config.getString("output.folder"),"meta_data");
        metaDataFolder.mkdirs();
        String[] trainIndexIds;

        trainIndexIds = getDocsForSplitFromQuery(index, config.getString("index.splitQuery.train"));



        LabelTranslator trainLabelTranslator = loadTrainLabelTranslator(config, index, trainIndexIds);
        Serialization.serialize(trainLabelTranslator,new File(metaDataFolder,"label_translator.ser"));
        FileUtils.writeStringToFile(new File(metaDataFolder,"label_translator.txt"),trainLabelTranslator.toString());

        FeatureList featureList = new FeatureList();
        if (config.getBoolean("feature.useInitialFeatures")){
            addInitialFeatures(config,index,featureList,trainIndexIds);
        }

        if (config.getBoolean("feature.useCodeDescription")){
            addCodeDescription(config, index, featureList);
        }


        Set<Ngram> ngrams = new HashSet<>();
        ngrams.addAll(gather(config,index,trainIndexIds));

        if (config.getBoolean("feature.filterNgramsByKeyWords")){
            ngrams = keywordsFilter(config,index,ngrams);
        }

        if (config.getBoolean("feature.filterNgramsByRegex")){
            ngrams = regexFilter(config,ngrams);
        }

        if (config.getBoolean("feature.addExternalNgrams")){
            ngrams.addAll(addNgramFromFile(config, index));
        }
//        if (config.getBoolean("feature.generateDistribution")){
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
        logger.info("meta data generated");


    }

    static void createDataSet(Config config, MultiLabelIndex index, String[] indexIds, String datasetName, String docFilter) throws Exception{
//        String splitValueAll = splitListToString(splitValues);


        logger.info("creating data set "+datasetName);
        File metaDataFolder = new File(config.getString("output.folder"),"meta_data");
        IdTranslator idTranslator = loadIdTranslator(indexIds);
        String archive = config.getString("output.folder");
        LabelTranslator trainLabelTranslator = (LabelTranslator)Serialization.deserialize(new File(metaDataFolder,"label_translator.ser"));
        LabelTranslator labelTranslator = loadAugmentedLabelTranslator(config, index, indexIds, trainLabelTranslator);

        FeatureList featureList = (FeatureList)Serialization.deserialize(new File(metaDataFolder,"feature_list.ser"));

        MultiLabelClfDataSet dataSet = loadData(config, index, featureList, idTranslator, featureList.size(), labelTranslator, docFilter);
        dataSet.setFeatureList(featureList);

        File dataFile = new File(new File(archive,"data_sets"),datasetName);

        TRECFormat.save(dataSet,dataFile);
        logger.info("data set "+datasetName+" created");

        ObjectMapper objectMapper = new ObjectMapper();
        objectMapper.writeValue(new File(dataFile,"data_config.json"),config);


    }

    static void createTrainSet(Config config, MultiLabelIndex index) throws Exception{
        generateMetaData(config, index);
        String[] indexIds = getDocsForSplitFromQuery(index, config.getString("index.splitQuery.train"));
        createDataSet(config, index, indexIds,config.getString("output.trainFolder"), config.getString("index.splitQuery.train"));
    }

    static void createTestSet(Config config, MultiLabelIndex index) throws Exception{
        String[] indexIds = getDocsForSplitFromQuery(index, config.getString("index.splitQuery.test"));
        createDataSet(config, index, indexIds,config.getString("output.testFolder"), config.getString("index.splitQuery.test"));
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
        String externalKeywordsFile = config.getString("feature.filterNgrams.keyWordsFile");
        List<String> lines = FileUtils.readLines(new File(externalKeywordsFile));
        String analyzer = config.getString("feature.analyzer");
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
        String regex = config.getString("feature.filterNgrams.regex");
        return ngrams.parallelStream().filter(ngram->!ngram.getNgram().matches(regex)).collect(Collectors.toSet());
    }



}
