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
import edu.neu.ccs.pyramid.feature_selection.FeatureDistribution;
import edu.neu.ccs.pyramid.util.Serialization;
import org.apache.commons.io.FileUtils;
import org.apache.commons.lang3.time.StopWatch;
import org.elasticsearch.search.aggregations.bucket.terms.Terms;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.util.*;
import java.util.regex.Pattern;
import java.util.stream.Collectors;

/**
 * for multi label dataset,
 * dump feature matrix with initial features and ngram features
 * follow exp12
 * Created by chengli on 6/12/15.
 */
public class App1 {
    public static void main(String[] args) throws Exception{
        if (args.length !=1){
            throw new IllegalArgumentException("Please specify a properties file.");
        }

        Config config = new Config(args[0]);
        System.out.println(config);
        File output = new File(config.getString("output.folder"));
        output.mkdirs();

        if (config.getBoolean("createTrainSet")){
            createTrainSet(config);
        }

        if (config.getBoolean("createTestSet")){
            createTestSet(config);
        }
    }

    public static void main(Config config) throws Exception{
        File output = new File(config.getString("output.folder"));
        output.mkdirs();

        if (config.getBoolean("createTrainSet")){
            createTrainSet(config);
        }

        if (config.getBoolean("createTestSet")){
            createTestSet(config);
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
        System.out.println("index loaded");
        System.out.println("there are "+index.getNumDocs()+" documents in the index.");
        return index;
    }

    static String[] getDocsForSplitFromField(Config config, ESIndex index, List<String> splitValues) throws Exception{
        String splitField = config.getString("index.splitField");
        Set<String> docs = new HashSet<>();
        for (String value: splitValues){
            docs.addAll(index.termFilter(splitField,value));
        }
        String[] ids = docs.toArray(new String[docs.size()]);
        return ids;
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
        System.out.println("all possible initial features:"+featureFields);

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
                    System.out.println("gathering "+n+ "-grams from field "+field+" with slop "+slop+" and minDf "+minDf);
                    NgramTemplate template = new NgramTemplate(field,n,slop);
                    Multiset<Ngram> ngrams = NgramEnumerator.gatherNgram(index, ids, template, minDf);
                    System.out.println("gathered "+ngrams.elementSet().size()+ " ngrams");
                    int newCounter = 0;
                    for (Multiset.Entry<Ngram> entry: ngrams.entrySet()){
                        Ngram ngram = entry.getElement();
                        int count = entry.getCount();
                        if (interesting(allNgrams,ngram,count)){
                            allNgrams.add(ngram,count);
                            newCounter += 1;
                        }
                    }
                    System.out.println(newCounter+" are really new");
                }
            }
        }
        System.out.println("there are "+allNgrams.elementSet().size()+" ngrams in total");
        BufferedWriter bufferedWriter = new BufferedWriter(new FileWriter(new File(metaDataFolder,"all_ngrams.txt")));
        for (Multiset.Entry<Ngram> ngramEntry: allNgrams.entrySet()){
            bufferedWriter.write(ngramEntry.getElement().toString());
            bufferedWriter.write("\t");
            bufferedWriter.write(""+ngramEntry.getCount());
            bufferedWriter.newLine();
        }

        bufferedWriter.close();

        //for serialization
        Set<Ngram> uniques = new HashSet<>();
        uniques.addAll(allNgrams.elementSet());
        Serialization.serialize(uniques, new File(metaDataFolder, "all_ngrams.ser"));
        return allNgrams.elementSet();
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
                                         LabelTranslator labelTranslator) throws Exception{
        int numDataPoints = idTranslator.numData();
        int numClasses = labelTranslator.getNumClasses();
        MultiLabelClfDataSet dataSet = MLClfDataSetBuilder.getBuilder()
                .numDataPoints(numDataPoints).numFeatures(totalDim)
                .numClasses(numClasses).dense(false)
                .missingValue(config.getBoolean("feature.missingValue")).build();
        for(int i=0;i<numDataPoints;i++){
            String dataIndexId = idTranslator.toExtId(i);
            List<String> extMultiLabel = index.getExtMultiLabel(dataIndexId);
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
            default:
                throw new IllegalArgumentException("unknown ngramMatchScoreType");
        }

        FeatureLoader.loadFeatures(index, dataSet, featureList, idTranslator, matchScoreType);

        dataSet.setIdTranslator(idTranslator);
        dataSet.setLabelTranslator(labelTranslator);
        return dataSet;
    }





    static LabelTranslator loadTrainLabelTranslator(Config config, MultiLabelIndex index, String[] trainIndexIds) throws Exception{
        Collection<Terms.Bucket> buckets = index.termAggregation(config.getString("index.labelField"), trainIndexIds);
        System.out.println("there are "+buckets.size()+" classes in the training set.");
        List<String> labels = new ArrayList<>();
        System.out.println("label distribution in training set:");
        for (Terms.Bucket bucket: buckets){
            System.out.print(bucket.getKey());
            System.out.print(":");
            System.out.print(bucket.getDocCount());
            System.out.print(", ");
            labels.add(bucket.getKey());
        }
        System.out.println();

        LabelTranslator labelTranslator = new LabelTranslator(labels);
        System.out.println(labelTranslator);
        return labelTranslator;
    }

    static LabelTranslator loadAugmentedLabelTranslator(Config config, MultiLabelIndex index, String[] testIndexIds, LabelTranslator trainLabelTranslator){
        List<String> extLabels = new ArrayList<>();
        for (int i=0;i<trainLabelTranslator.getNumClasses();i++){
            extLabels.add(trainLabelTranslator.toExtLabel(i));
        }

        Collection<Terms.Bucket> buckets = index.termAggregation(config.getString("index.labelField"), testIndexIds);
        List<String> newLabels = new ArrayList<>();
        System.out.println("label distribution in data set:");
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
        System.out.println();
        if (!newLabels.isEmpty()){
            System.out.println("WARNING: found new labels in data set: "+newLabels);
        }
        return new LabelTranslator(extLabels);
    }









    static void getNgramDistributions(Config config, ESIndex index, String[] ids,LabelTranslator labelTranslator ) throws Exception{
        File metaDataFolder = new File(config.getString("output.folder"),"meta_data");
        metaDataFolder.mkdirs();

        System.out.println("generating ngram distributions");
        StopWatch stopWatch = new StopWatch();
        stopWatch.start();
        File file = new File(metaDataFolder,"all_ngrams.ser");
        Set<Ngram> ngrams= (Set) Serialization.deserialize(file);
        String labelField = config.getString("index.labelField");
        long[] labelDistribution = LabelDistribution.getLabelDistribution(index,labelField,ids,labelTranslator);
        List<FeatureDistribution> distributions = ngrams.stream().parallel()
                .map(ngram -> new FeatureDistribution(ngram, index, labelField, ids, labelTranslator,labelDistribution))
                .collect(Collectors.toList());
        Serialization.serialize(distributions,new File(metaDataFolder,"distributions.ser"));
        BufferedWriter bufferedWriter = new BufferedWriter(new FileWriter(new File(metaDataFolder,"distributions.txt")));
        for (FeatureDistribution distribution: distributions){
            bufferedWriter.write(distribution.toString());
            bufferedWriter.newLine();
        }

        bufferedWriter.close();
        System.out.println("done");
        System.out.println("time spent on generating distributions = "+stopWatch);
    }
    
    static void generateMetaData(Config config) throws Exception{
        System.out.println("generating meta data");
        File metaDataFolder = new File(config.getString("output.folder"),"meta_data");
        metaDataFolder.mkdirs();
        MultiLabelIndex index = loadIndex(config);
        String[] trainIndexIds;
        String splitMode = config.getString("index.splitMode");
        switch (splitMode) {
            case "field":
                trainIndexIds = getDocsForSplitFromField(config, index, config.getStrings("index.splitField.train"));
                break;
            case "query":
                trainIndexIds = getDocsForSplitFromQuery(index, config.getString("index.splitQuery.train"));
                break;
            default:
                throw new IllegalArgumentException("unknown split mode");
        }


        LabelTranslator trainLabelTranslator = loadTrainLabelTranslator(config, index, trainIndexIds);
        Serialization.serialize(trainLabelTranslator,new File(metaDataFolder,"label_translator.ser"));
        FileUtils.writeStringToFile(new File(metaDataFolder,"label_translator.txt"),trainLabelTranslator.toString());

        FeatureList featureList = new FeatureList();
        if (config.getBoolean("feature.useInitialFeatures")){
            addInitialFeatures(config,index,featureList,trainIndexIds);
        }

        Set<Ngram> ngrams = gather(config,index,trainIndexIds);
        getNgramDistributions(config,index,trainIndexIds,trainLabelTranslator);
        addNgramFeatures(featureList,ngrams);

        Serialization.serialize(featureList,new File(metaDataFolder,"feature_list.ser"));
        try (BufferedWriter bufferedWriter = new BufferedWriter(new FileWriter(new File(metaDataFolder,"feature_list.txt")))
        ){
            for (Feature feature: featureList.getAll()){
                bufferedWriter.write(feature.toString());
                bufferedWriter.newLine();
            }
        }

        index.close();
        System.out.println("meta data generated");
    }

    static void createDataSet(Config config, String[] indexIds, String datasetName) throws Exception{
//        String splitValueAll = splitListToString(splitValues);


        System.out.println("creating data set "+datasetName);
        File metaDataFolder = new File(config.getString("output.folder"),"meta_data");
        MultiLabelIndex index = loadIndex(config);
        IdTranslator idTranslator = loadIdTranslator(indexIds);
        String archive = config.getString("output.folder");
        LabelTranslator trainLabelTranslator = (LabelTranslator)Serialization.deserialize(new File(metaDataFolder,"label_translator.ser"));
        LabelTranslator labelTranslator = loadAugmentedLabelTranslator(config, index, indexIds, trainLabelTranslator);

        FeatureList featureList = (FeatureList)Serialization.deserialize(new File(metaDataFolder,"feature_list.ser"));

        MultiLabelClfDataSet dataSet = loadData(config, index, featureList, idTranslator, featureList.size(), labelTranslator);
        dataSet.setFeatureList(featureList);

        File dataFile = new File(new File(archive,"data_sets"),datasetName);

        TRECFormat.save(dataSet,dataFile);
        index.close();
        System.out.println("data set "+datasetName+" created");

        ObjectMapper objectMapper = new ObjectMapper();
        objectMapper.writeValue(new File(dataFile,"data_config.json"),config);

    }

    static void createTrainSet(Config config) throws Exception{
        generateMetaData(config);
        String[] indexIds;
        String splitMode = config.getString("index.splitMode");
        ESIndex index =  loadIndex(config);
        switch (splitMode) {
            case "field":
                indexIds = getDocsForSplitFromField(config, index, config.getStrings("index.splitField.train"));
                break;
            case "query":
                indexIds = getDocsForSplitFromQuery(index, config.getString("index.splitQuery.train"));
                break;
            default:
                throw new IllegalArgumentException("unknown split mode");
        }
        createDataSet(config,indexIds,"train");
    }

    static void createTestSet(Config config) throws Exception{
        String[] indexIds;
        String splitMode = config.getString("index.splitMode");
        ESIndex index =  loadIndex(config);
        switch (splitMode) {
            case "field":
                indexIds = getDocsForSplitFromField(config, index, config.getStrings("index.splitField.test"));
                break;
            case "query":
                indexIds = getDocsForSplitFromQuery(index, config.getString("index.splitQuery.test"));
                break;
            default:
                throw new IllegalArgumentException("unknown split mode");
        }
        createDataSet(config,indexIds,"test");
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





}
