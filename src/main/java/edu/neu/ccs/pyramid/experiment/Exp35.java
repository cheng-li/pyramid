package edu.neu.ccs.pyramid.experiment;

import edu.neu.ccs.pyramid.configuration.Config;
import edu.neu.ccs.pyramid.dataset.*;
import edu.neu.ccs.pyramid.elasticsearch.ESIndex;
import edu.neu.ccs.pyramid.elasticsearch.FeatureLoader;
import edu.neu.ccs.pyramid.elasticsearch.SingleLabelIndex;
import edu.neu.ccs.pyramid.elasticsearch.TermStat;
import edu.neu.ccs.pyramid.feature.*;
import edu.neu.ccs.pyramid.feature_extraction.NgramEnumerator;
import edu.neu.ccs.pyramid.util.Sampling;
import org.apache.commons.io.FileUtils;
import org.elasticsearch.action.search.SearchResponse;
import org.elasticsearch.search.SearchHit;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.*;
import java.util.concurrent.ConcurrentHashMap;
import java.util.regex.Pattern;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

/**
 * dump ngram features
 * split to train/valid/test
 * Created by chengli on 12/19/14.
 *
 */
public class Exp35 {
    public static void main(String[] args) throws Exception{
        if (args.length !=1){
            throw new IllegalArgumentException("please specify the config file");
        }

        Config config = new Config(args[0]);
        System.out.println(config);

        SingleLabelIndex index = loadIndex(config);
        build(config,index);
        index.close();
    }

    static SingleLabelIndex loadIndex(Config config) throws Exception{
        SingleLabelIndex.Builder builder = new SingleLabelIndex.Builder()
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
        SingleLabelIndex index = builder.build();
        System.out.println("index loaded");
        System.out.println("there are "+index.getNumDocs()+" documents in the index.");
//        for (int i=0;i<index.getNumDocs();i++){
//            System.out.println(i);
//            System.out.println(index.getLabel(""+i));
//        }
        return index;
    }

    static String[] sampleTrain(Config config, SingleLabelIndex index, Set<String> duplicate){
        int numDocsInIndex = index.getNumDocs();
        String[] ids = null;

        String splitField = config.getString("index.splitField");
        List<String> train = IntStream.range(0, numDocsInIndex).parallel()
                .filter(i -> index.getStringField("" + i, splitField).
                        equalsIgnoreCase("train")).
                        mapToObj(i -> "" + i).filter(id -> !duplicate.contains(id)).collect(Collectors.toList());
        ids = train.toArray(new String[train.size()]);
        return ids;
    }

    static String[] sampleTest(Config config, SingleLabelIndex index){
        int numDocsInIndex = index.getNumDocs();
        String[] ids = null;

        String splitField = config.getString("index.splitField");
        ids = IntStream.range(0, numDocsInIndex).parallel().
                filter(i -> index.getStringField("" + i, splitField).
                        equalsIgnoreCase("test")).
                mapToObj(i -> "" + i).collect(Collectors.toList()).
                toArray(new String[0]);
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
     * @param ids pull features from train ids
     * @throws Exception
     */
    static void addInitialFeatures(Config config, SingleLabelIndex index, List<Feature> features,
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
                expander.setStart(features.size());
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
                    features.addAll(group);
                }

            } else {
                Feature feature = new Feature();
                feature.setName(field);
                feature.setIndex(features.size());
                feature.getSettings().put("source","field");
                features.add(feature);
            }
        }

    }



    static List<String> gather(Config config, SingleLabelIndex index, String field,
                               String[] ids) throws Exception{
        List<Integer> ns = config.getIntegers("ngram.n");
        List<Integer> minDfs = config.getIntegers("ngram.minDf");
        List<String> list = new ArrayList<>();
        for (int i=0;i<ns.size();i++){
            int n = ns.get(i);
            int minDf = minDfs.get(i);
            if (n==1){
                list.addAll(gatherUnigrams(index,field, ids,minDf));
            } else {
                list.addAll(gatherNgrams(index, field,ids, n, minDf));
            }
        }
        return list;
    }

    static List<String> gatherUnigrams(ESIndex index, String field,
                                       String[] ids, int minDf) throws Exception{
        System.out.println("gathering unigrams with minDf "+minDf+" from field "+field);
        Set<TermStat> unigrams = Collections.newSetFromMap(new ConcurrentHashMap<TermStat, Boolean>());
        Arrays.stream(ids).parallel().forEach(id -> {
            Set<TermStat> termStats = null;
            try {
                termStats = index.getTermStats(field, id);
            } catch (IOException e) {
                System.out.println("id= "+id);
                e.printStackTrace();
            }
            termStats.stream().filter(termStat -> termStat.getDf() > minDf).forEach(unigrams::add);
        });

        List<String> list = unigrams.stream().sorted(Comparator.comparing(TermStat::getTerm))
                .sorted(Comparator.comparing(TermStat::getDf).reversed())
                .map(TermStat::getTerm)
                .collect(Collectors.toList());
        System.out.println("done");
        System.out.println("there are "+list.size()+" unigrams");
        return list;
    }

    static List<String> gatherNgrams(ESIndex index, String field,
                                       String[] ids, int n, int minDf) throws Exception{

        System.out.println("gathering "+n+"-grams with minDf "+minDf+" from field "+field);
        List<String> ngrams = NgramEnumerator.gatherNgrams(index,field,ids,n,minDf);
        System.out.println("done");
        System.out.println("there are "+ngrams.size()+" "+n+"-grams");
        return ngrams;
    }

    static void addNgramFeatures(List<Feature> features, List<String> ngrams, String field, int slop){
        for (String ngram: ngrams){
            String featureName = ngram+"(slop="+slop+")";
            Feature feature = new Feature();
            feature.setIndex(features.size());
            feature.setName(featureName);
            feature.getSettings().put("ngram",ngram);
            feature.getSettings().put("source","matching_score");
            feature.getSettings().put("slop",""+slop);
            feature.getSettings().put("field",field);
            features.add(feature);
        }
    }

    static ClfDataSet loadData(Config config, SingleLabelIndex index,
                               List<Feature> features,
                               IdTranslator idTranslator, int totalDim,
                               LabelTranslator labelTranslator) throws Exception{
        int numDataPoints = idTranslator.numData();
        int numClasses = config.getInt("numClasses");
        ClfDataSet dataSet = ClfDataSetBuilder.getBuilder()
                .numDataPoints(numDataPoints).numFeatures(totalDim)
                .numClasses(numClasses).dense(!config.getBoolean("featureMatrix.sparse"))
                .missingValue(config.getBoolean("featureMatrix.missingValue"))
                .build();

        IntStream.range(0,numDataPoints).parallel()
                .forEach(i -> {
                    String dataIndexId = idTranslator.toExtId(i);
                    int label = index.getLabel(dataIndexId);
                    dataSet.setLabel(i,label);
                });

        FeatureLoader.loadFeatures(index,dataSet,features,idTranslator);

        DataSetUtil.setIdTranslator(dataSet, idTranslator);
        DataSetUtil.setLabelTranslator(dataSet, labelTranslator);
        return dataSet;
    }

    static ClfDataSet loadTrainData(Config config, SingleLabelIndex index, List<Feature> features,
                                    IdTranslator idTranslator, LabelTranslator labelTranslator) throws Exception{
        System.out.println("creating training set");
        int totalDim = features.size();
        System.out.println("allocating "+totalDim+" columns for training set");
        ClfDataSet dataSet = loadData(config,index,features,idTranslator,totalDim,labelTranslator);
        System.out.println("training set created");
        return dataSet;
    }

    static ClfDataSet loadTestData(Config config, SingleLabelIndex index,
                                   List<Feature> features, IdTranslator idTranslator,
                                   LabelTranslator labelTranslator) throws Exception{
        System.out.println("creating test set");

        int totalDim = features.size();

        ClfDataSet dataSet = loadData(config,index,features,idTranslator,totalDim,labelTranslator);
        System.out.println("test set created");
        return dataSet;
    }

    static ClfDataSet loadValidData(Config config, SingleLabelIndex index,
                                   FeatureMappers featureMappers, IdTranslator idTranslator,
                                   LabelTranslator labelTranslator) throws Exception{
        System.out.println("creating validation set");

        int totalDim = featureMappers.getTotalDim();

        ClfDataSet dataSet = loadData(config,index,featureMappers,idTranslator,totalDim,labelTranslator);
        System.out.println("validation set created");
        return dataSet;
    }

    //todo speed up and only look at train
    static LabelTranslator loadLabelTranslator(Config config, SingleLabelIndex index) throws Exception{
        System.out.println("loading label translator...");
        int numClasses = config.getInt("numClasses");
        int numDocs = index.getNumDocs();
        Map<Integer, String> map = new ConcurrentHashMap<>();
        while(map.size()<numClasses){
            int i = Sampling.intUniform(0,numDocs-1);
            int intLabel = index.getLabel(""+i);
            String extLabel = index.getExtLabel("" + i);
            map.put(intLabel,extLabel);
        }
        System.out.println("loaded");
        return new LabelTranslator(map);
    }

    static void showDistribution(Config config, ClfDataSet dataSet, LabelTranslator labelTranslator){
        int numClasses = config.getInt("numClasses");
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
        DataSetUtil.dumpDataPointSettings(dataSet, new File(dataFile, "data_settings.txt"));
        DataSetUtil.dumpFeatureSettings(dataSet,new File(dataFile,"feature_settings.txt"));
        System.out.println("data set saved to "+dataFile.getAbsolutePath());
    }

    static void dumpTrainFeatures(Config config, SingleLabelIndex index, IdTranslator idTranslator) throws Exception{
        String archive = config.getString("archive.folder");
        String trecFile = new File(archive,config.getString("archive.trainingSet")).getAbsolutePath();
        String file = new File(trecFile,"dumped_fields.txt").getAbsolutePath();
        dumpFeatures(config,index,idTranslator,file);
    }

    static void dumpTestFeatures(Config config, SingleLabelIndex index, IdTranslator idTranslator) throws Exception{
        String archive = config.getString("archive.folder");
        String trecFile = new File(archive,config.getString("archive.testSet")).getAbsolutePath();
        String file = new File(trecFile,"dumped_fields.txt").getAbsolutePath();
        dumpFeatures(config,index,idTranslator,file);
    }

    static void dumpFeatures(Config config, SingleLabelIndex index, IdTranslator idTranslator, String fileName) throws Exception{

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

    static void build(Config config, SingleLabelIndex index) throws Exception{
        int numDocsInIndex = index.getNumDocs();
        Set<String> duplidate = loadDuplicate(config);
        String[] trainIndexIds = sampleTrain(config,index,duplidate);
        System.out.println("number of training documents = "+trainIndexIds.length);
        IdTranslator trainIdTranslator = loadIdTranslator(trainIndexIds);
        List<Feature> features = new ArrayList<>();

        LabelTranslator labelTranslator = loadLabelTranslator(config, index);
        if (config.getBoolean("useInitialFeatures")){
            addInitialFeatures(config,index,,features,trainIndexIds);
        }

        List<String> fields = config.getStrings("fields");
        List<Integer> slops = config.getIntegers("ngram.slop");
        for (String field: fields){
            List<String> ngrams = gather(config,index,field, trainIndexIds);
            for (int slop: slops){
                addNgramFeatures(features, ngrams,field, slop);
            }
        }





        ClfDataSet trainDataSet = loadTrainData(config,index,features, trainIdTranslator, labelTranslator);
        System.out.println("in training set :");
        showDistribution(config,trainDataSet,labelTranslator);


        DataSetUtil.setFeatureMappers(trainDataSet,featureMappers);
        saveDataSet(config, trainDataSet, config.getString("archive.trainingSet"));
        if (config.getBoolean("archive.dumpFields")){
            dumpTrainFeatures(config,index,trainIdTranslator);
        }

        String[] testIndexIds = sampleTest(config,index);
        IdTranslator testIdTranslator = loadIdTranslator(testIndexIds);

        ClfDataSet testDataSet = loadTestData(config,index,featureMappers,testIdTranslator,labelTranslator);
        DataSetUtil.setFeatureMappers(testDataSet,featureMappers);
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
