package edu.neu.ccs.pyramid.experiment;

import com.google.common.collect.ConcurrentHashMultiset;
import com.google.common.collect.Multiset;
import edu.neu.ccs.pyramid.configuration.Config;
import edu.neu.ccs.pyramid.dataset.*;
import edu.neu.ccs.pyramid.elasticsearch.ESIndex;
import edu.neu.ccs.pyramid.elasticsearch.FeatureLoader;
import edu.neu.ccs.pyramid.feature.*;
import edu.neu.ccs.pyramid.feature_extraction.NgramEnumerator;
import edu.neu.ccs.pyramid.feature_extraction.NgramTemplate;
import edu.neu.ccs.pyramid.feature_selection.FusedKolmogorovFilter;
import edu.neu.ccs.pyramid.feature_selection.LRGradientSelection;
import edu.neu.ccs.pyramid.feature_selection.NgramClassDistribution;
import edu.neu.ccs.pyramid.sentiment_analysis.Negation;
import edu.neu.ccs.pyramid.util.Serialization;
import org.apache.commons.io.FileUtils;
import org.apache.commons.lang3.time.StopWatch;
import org.elasticsearch.search.aggregations.bucket.terms.Terms;

import java.io.*;
import java.util.*;
import java.util.concurrent.ConcurrentHashMap;
import java.util.regex.Pattern;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

/**
 * spanNot query based on bigrams
 * following exp35
 * Created by chengli on 4/26/15.
 */
public class Exp88 {
    public static void main(String[] args) throws Exception{
        if (args.length !=1){
            throw new IllegalArgumentException("please specify the config file");
        }

        Config config = new Config(args[0]);
        System.out.println(config);
        File output = new File(config.getString("archive.folder"));
        output.mkdirs();

        ESIndex index = loadIndex(config);
        build(config,index);
        index.close();
    }

    static ESIndex loadIndex(Config config) throws Exception{
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
        System.out.println("index loaded");
        System.out.println("there are "+index.getNumDocs()+" documents in the index.");
//        for (int i=0;i<index.getNumDocs();i++){
//            System.out.println(i);
//            System.out.println(index.getLabel(""+i));
//        }
        return index;
    }

    static String[] sampleTrain(Config config, ESIndex index, Set<String> duplicate){
        int numDocsInIndex = index.getNumDocs();
        String[] ids = null;

        if (config.getString("split.fashion").equalsIgnoreCase("fixed")){
            String splitField = config.getString("index.splitField");
            List<String> train = IntStream.range(0, numDocsInIndex).parallel()
                    .filter(i -> index.getStringField("" + i, splitField).
                            equalsIgnoreCase("train")).
                            mapToObj(i -> "" + i).filter(id -> !duplicate.contains(id)).collect(Collectors.toList());
            ids = train.toArray(new String[train.size()]);
        } else if (config.getString("split.fashion").equalsIgnoreCase("random")){
            int numFolds = config.getInt("split.random.numFolds");
            ids = IntStream.range(0, numDocsInIndex).parallel()
                    //todo make a parameter?
                    .filter(i -> i % numFolds != 0).mapToObj(i -> ""+i).toArray(String[]::new);
        } else {
            throw new RuntimeException("illegal split fashion");
        }

        return ids;
    }

    static String[] sampleTest(Config config, ESIndex index) {
        int numDocsInIndex = index.getNumDocs();
        String[] ids = null;

        if (config.getString("split.fashion").equalsIgnoreCase("fixed")) {
            String splitField = config.getString("index.splitField");
            List<String> list = IntStream.range(0, numDocsInIndex).parallel()
                    .filter(i -> index.getStringField("" + i, splitField).
                            equalsIgnoreCase("test")).
                            mapToObj(i -> "" + i).collect(Collectors.toList());
            ids = list.toArray(new String[list.size()]);
        } else if (config.getString("split.fashion").equalsIgnoreCase("random")) {
            int numFolds = config.getInt("split.random.numFolds");
            ids = IntStream.range(0, numDocsInIndex).parallel()
                    //todo make a parameter?
                    .filter(i -> i % numFolds == 0).mapToObj(i -> "" + i).toArray(String[]::new);
        } else {
            throw new RuntimeException("illegal split fashion");
        }

        return ids;
    }

//    static String[] sampleValid(Config config, SingleLabelIndex index){
//        int numDocsInIndex = index.getNumDocs();
//        String[] ids = null;
//
//        String splitField = config.getString("index.splitField");
//        ids = IntStream.range(0, numDocsInIndex).parallel().
//                filter(i -> index.getStringField("" + i, splitField).
//                        equalsIgnoreCase("valid")).
//                mapToObj(i -> "" + i).collect(Collectors.toList()).
//                toArray(new String[0]);
//        return ids;
//    }

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

    /**
     *
     * @param config
     * @param index
     * @param ids pull featureList from train ids
     * @throws Exception
     */
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
                System.out.println("expanding categorical feature "+field);
                CategoricalFeatureExpander expander = new CategoricalFeatureExpander();
                expander.setStart(featureList.size());
                expander.setVariableName(field);
                expander.putSetting("source","field");
                Set<String> categories = Collections.newSetFromMap(new ConcurrentHashMap<String, Boolean>());
                Arrays.stream(ids).parallel().forEach(id -> {
                    String category = index.getStringField(id, field);
                    categories.add(category);
                });
                for (String category: categories){
                    expander.addCategory(category);
                }
//                for (String id: ids){
//                    String category = index.getStringField(id, field);
//                    expander.addCategory(category);
//                }
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
                feature.getSettings().put("source", "field");
                featureList.add(feature);
            }
        }

    }


    /**
     * not interesting if essentially the same as an ngram with smaller slop (counts are the same)
     * interesting if count for bigger slop > count for smaller slop
     * @param allNgrams
     * @param candidate
     * @param count
     * @return
     */
    static boolean interesting(Multiset<Ngram> allNgrams, Ngram candidate, int count){
        if (allNgrams.contains(candidate)){
            return false;
        }
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
        Multiset<Ngram> allNgrams = ConcurrentHashMultiset.create();
        List<Integer> ns = config.getIntegers("ngram.n");
        int minDf = config.getInt("ngram.minDf");
        List<String> fields = config.getStrings("fields");
        if (config.getBoolean("useNounField")){
            Set<String> all = index.listAllFields();
            for (String field: all){
                if (field.startsWith("noun_")){
                    fields.add(field);
                }
            }
        }
        System.out.println("fields to be considered = "+fields);

        List<Integer> slops = config.getIntegers("ngram.slop");
        for (String field: fields){
            for (int n: ns){
                for (int slop: slops){
//                    if (n==1 && slop>0){
//                        continue;
//                    }
                    System.out.println("gathering "+n+ "-grams from field "+field+" with slop "+slop+" and minDf "+minDf);
                    NgramTemplate template = new NgramTemplate(field,n,slop);
                    Multiset<Ngram> ngrams = NgramEnumerator.gatherNgram(index, ids, template, minDf);
                    System.out.println("gathered "+ngrams.elementSet().size()+ " ngrams");

                    int newCounter = 0;
                    for (Multiset.Entry<Ngram> entry: ngrams.entrySet()){
                        Ngram ngram = entry.getElement();
                        int count = entry.getCount();
                        boolean cond1 = interesting(allNgrams,ngram,count);
                        boolean condition = cond1;
                        if (condition){
                            allNgrams.add(ngram,count);
                            newCounter += 1;
                        }
                    }
                    System.out.println(newCounter+" are really new");
                }
            }
        }




        for (String field: fields){
            for (int n: config.getIntegers("negation.n")){
                for (int slop: config.getIntegers("negation.slop")){

                    System.out.println("gathering "+n+ "-grams with negations from field "+field+" with slop "+slop+" and minDf "+minDf);
                    NgramTemplate template = new NgramTemplate(field,n,slop);
                    Multiset<Ngram> ngrams = NgramEnumerator.gatherNgram(index,ids,template,minDf);

                    int newCounter = 0;
                    for (Multiset.Entry<Ngram> entry: ngrams.entrySet()){
                        Ngram ngram = entry.getElement();
                        int count = entry.getCount();
                        boolean cond1 = interesting(allNgrams,ngram,count);
                        boolean cond2 =  Negation.containsNegation(ngram.getNgram());
                        boolean condition = cond1 && cond2;
                        if (condition){
                            allNgrams.add(ngram,count);
                            newCounter += 1;
                        }
                    }
                    System.out.println(newCounter+" are really new");
                }
            }
        }

        System.out.println("there are "+allNgrams.elementSet().size()+" ngrams in total");
        BufferedWriter bufferedWriter = new BufferedWriter(new FileWriter(new File(config.getString("archive.folder"),"allFeatures.txt")));
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
        Serialization.serialize(uniques, new File(config.getString("archive.folder"), "allFeatures.ser"));
        return allNgrams.elementSet();
    }






    static void addNgramFeatures(Config config, FeatureList featureList) throws Exception{

        File file = new File(config.getString("archive.folder"),"allFeatures.ser");
        Collection<Ngram> ngrams;
        try(
                FileInputStream fileInputStream = new FileInputStream(file);
                BufferedInputStream bufferedInputStream = new BufferedInputStream(fileInputStream);
                ObjectInputStream objectInputStream = new ObjectInputStream(bufferedInputStream);
        ){
            ngrams = (Collection)objectInputStream.readObject();
        }

        ngrams.stream().forEach(ngram -> {
            ngram.getSettings().put("source", "matching_score");
            featureList.add(ngram);
        });
    }


    static void addSpanNotNgramFeatures(Config config, FeatureList featureList) throws Exception{

        File file = new File(config.getString("archive.folder"),"allFeatures.ser");
        Collection<Ngram> ngrams;
        try(
                FileInputStream fileInputStream = new FileInputStream(file);
                BufferedInputStream bufferedInputStream = new BufferedInputStream(fileInputStream);
                ObjectInputStream objectInputStream = new ObjectInputStream(bufferedInputStream);
        ){
            ngrams = (Collection)objectInputStream.readObject();
        }


        ngrams.stream().filter(ngram -> ngram.getN()==2)
                .forEach(ngram -> {
                    List<SpanNotNgram> spanNotNgrams = SpanNotNgram.breakBigram(ngram);

                    for (SpanNotNgram spanNotNgram: spanNotNgrams){
                        featureList.add(spanNotNgram);
                    }
                });
    }

    static ClfDataSet loadData(Config config, ESIndex index,
                               FeatureList featureList,
                               IdTranslator idTranslator, int totalDim,
                               LabelTranslator labelTranslator) throws Exception{
        int numDataPoints = idTranslator.numData();
        int numClasses = labelTranslator.getNumClasses();
        ClfDataSet dataSet = ClfDataSetBuilder.getBuilder()
                .numDataPoints(numDataPoints).numFeatures(totalDim)
                .numClasses(numClasses).dense(!config.getBoolean("featureMatrix.sparse"))
                .missingValue(config.getBoolean("featureMatrix.missingValue"))
                .build();
        String labelField = config.getString("index.labelField");
        IntStream.range(0,numDataPoints).parallel()
                .forEach(i -> {
                    String dataIndexId = idTranslator.toExtId(i);
                    int label = labelTranslator.toIntLabel(index.getStringField(dataIndexId, labelField));
                    dataSet.setLabel(i,label);
                });

        FeatureLoader.loadFeatures(index,dataSet,featureList,idTranslator);

        dataSet.setIdTranslator(idTranslator);
        dataSet.setLabelTranslator(labelTranslator);
        return dataSet;
    }

    static ClfDataSet loadTrainData(Config config, ESIndex index, FeatureList features,
                                    IdTranslator idTranslator, LabelTranslator labelTranslator) throws Exception{
        System.out.println("creating training set");
        int totalDim = features.size();
        System.out.println("allocating "+totalDim+" columns for training set");
        ClfDataSet dataSet = loadData(config,index,features,idTranslator,totalDim,labelTranslator);
        System.out.println("training set created");
        return dataSet;
    }

    static ClfDataSet loadTestData(Config config, ESIndex index,
                                   FeatureList features, IdTranslator idTranslator,
                                   LabelTranslator labelTranslator) throws Exception{
        System.out.println("creating test set");

        int totalDim = features.size();

        ClfDataSet dataSet = loadData(config,index,features,idTranslator,totalDim,labelTranslator);
        System.out.println("test set created");
        return dataSet;
    }

//    static ClfDataSet loadValidData(Config config, SingleLabelIndex index,
//                                   FeatureMappers featureMappers, IdTranslator idTranslator,
//                                   LabelTranslator labelTranslator) throws Exception{
//        System.out.println("creating validation set");
//
//        int totalDim = featureMappers.getTotalDim();
//
//        ClfDataSet dataSet = loadData(config,index,featureMappers,idTranslator,totalDim,labelTranslator);
//        System.out.println("validation set created");
//        return dataSet;
//    }


    static LabelTranslator loadLabelTranslator(Config config,ESIndex index,String[] trainIndexIds) throws Exception{
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

    static void showDistribution(ClfDataSet dataSet, LabelTranslator labelTranslator){
        int numClasses = dataSet.getNumClasses();
        int[] counts = new int[numClasses];
        int[] labels = dataSet.getLabels();
        for (int i=0;i<dataSet.getNumDataPoints();i++){
            int label = labels[i];
            counts[label] += 1;

        }
        System.out.println("label distribution:");
        for (int i=0;i<numClasses;i++){
            System.out.print(i+"("+labelTranslator.toExtLabel(i)+"):"+counts[i]+", ");
        }
        System.out.println("");
    }

    static void saveDataSet(Config config, ClfDataSet dataSet, String name) throws Exception{
        String archive = config.getString("archive.folder");
        File dataFile = new File(archive,name);
        TRECFormat.save(dataSet, dataFile);
        System.out.println("data set saved to "+dataFile.getAbsolutePath());
    }

    static void dumpTrainFeatures(Config config, ESIndex index, IdTranslator idTranslator) throws Exception{
        String archive = config.getString("archive.folder");
        String trecFile = new File(archive,config.getString("archive.trainingSet")).getAbsolutePath();
        String file = new File(trecFile,"dumped_fields.txt").getAbsolutePath();
        dumpFeatures(config,index,idTranslator,file);
    }

    static void dumpTestFeatures(Config config, ESIndex index, IdTranslator idTranslator) throws Exception{
        String archive = config.getString("archive.folder");
        String trecFile = new File(archive,config.getString("archive.testSet")).getAbsolutePath();
        String file = new File(trecFile,"dumped_fields.txt").getAbsolutePath();
        dumpFeatures(config,index,idTranslator,file);
    }

    static void dumpFeatures(Config config, ESIndex index, IdTranslator idTranslator, String fileName) throws Exception{

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

    static void build(Config config, ESIndex index) throws Exception{
        Set<String> duplidate = loadDuplicate(config);
        String[] trainIndexIds = sampleTrain(config,index,duplidate);
        System.out.println("number of training documents = "+trainIndexIds.length);
        IdTranslator trainIdTranslator = loadIdTranslator(trainIndexIds);
        FeatureList featureList = new FeatureList();

        LabelTranslator labelTranslator = loadLabelTranslator(config, index, trainIndexIds);
        if (config.getBoolean("useInitialFeatures")){
            addInitialFeatures(config,index,featureList,trainIndexIds);
        }

        if (config.getBoolean("gather")){
            gather(config,index,trainIndexIds);
        }




        addNgramFeatures(config, featureList);

        addSpanNotNgramFeatures(config,featureList);

        ClfDataSet trainDataSet = loadTrainData(config,index,featureList, trainIdTranslator, labelTranslator);
        System.out.println("in training set :");
        showDistribution(trainDataSet,labelTranslator);

        trainDataSet.setFeatureList(featureList);

        saveDataSet(config, trainDataSet, config.getString("archive.trainingSet"));
        if (config.getBoolean("archive.dumpFields")){
            dumpTrainFeatures(config,index,trainIdTranslator);
        }

        String[] testIndexIds = sampleTest(config,index);
        IdTranslator testIdTranslator = loadIdTranslator(testIndexIds);

        //todo new labels in test?
        ClfDataSet testDataSet = loadTestData(config,index,featureList,testIdTranslator,labelTranslator);
        testDataSet.setFeatureList(featureList);
        saveDataSet(config, testDataSet, config.getString("archive.testSet"));

//        String[] validIndexIds = sampleValid(config,index);
//        IdTranslator validIdTranslator = loadIdTranslator(validIndexIds);
//
//        ClfDataSet validDataSet = loadValidData(config,index,featureMappers,validIdTranslator,labelTranslator);
//        DataSetUtil.setFeatureMappers(validDataSet,featureMappers);
//        saveDataSet(config, validDataSet, config.getString("archive.validSet"));

        if (config.getBoolean("archive.dumpFields")){
            dumpTestFeatures(config,index,testIdTranslator);
        }
    }

    static Set<String> loadDuplicate(Config config) throws Exception{
        Set<String> set = new HashSet<>();
        if (config.getString("input.duplicate").equals("")){
            return set;
        }
        File file = new File(config.getString("input.duplicate"));
        String[] strArr = FileUtils.readFileToString(file).split(",");

        Arrays.stream(strArr).forEach(set::add);
        return set;
    }

}
