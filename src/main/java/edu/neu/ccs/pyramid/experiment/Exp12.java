package edu.neu.ccs.pyramid.experiment;

import com.google.common.collect.ConcurrentHashMultiset;
import com.google.common.collect.Multiset;
import edu.neu.ccs.pyramid.configuration.Config;
import edu.neu.ccs.pyramid.dataset.*;
import edu.neu.ccs.pyramid.elasticsearch.*;
import edu.neu.ccs.pyramid.feature.*;
import edu.neu.ccs.pyramid.feature_extraction.NgramEnumerator;
import edu.neu.ccs.pyramid.feature_extraction.NgramTemplate;
import edu.neu.ccs.pyramid.util.Sampling;
import org.elasticsearch.action.search.SearchResponse;
import org.elasticsearch.search.SearchHit;
import org.elasticsearch.search.aggregations.bucket.terms.Terms;

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
 * for multi label dataset,
 * dump feature matrix with initial features and ngram features
 * follow exp11
 * Created by chengli on 10/11/14.
 */
public class Exp12 {
    public static void main(String[] args) throws Exception{
        if (args.length !=1){
            throw new IllegalArgumentException("please specify the config file");
        }

        Config config = new Config(args[0]);
        System.out.println(config);
        File output = new File(config.getString("archive.folder"));
        output.mkdirs();

        MultiLabelIndex index = loadIndex(config);
        build(config,index);
        index.close();
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
            trainIds = Arrays.stream(Sampling.sampleByPercentage(numDocsInIndex,config.getDouble("split.random.trainPercentage"))).
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
        System.out.println("all possible initial features:"+featureFields);

        for (String field: featureFields){
            String featureType = index.getFieldType(field);
            if (featureType.equalsIgnoreCase("string")){
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
                feature.getSettings().put("source","field");
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
        Multiset<Ngram> allNgrams = ConcurrentHashMultiset.create();
        List<Integer> ns = config.getIntegers("ngram.n");
        int minDf = config.getInt("ngram.minDf");
        List<String> fields = config.getStrings("fields");
        List<Integer> slops = config.getIntegers("ngram.slop");
        for (String field: fields){
            for (int n: ns){
                for (int slop:slops){
                    if (n==1 && slop>0){
                        continue;
                    }
                    System.out.println("gathering "+n+ "-grams from field "+field+" with slop "+slop+" and minDf "+minDf);
                    NgramTemplate template = new NgramTemplate(field,n,slop);
                    Multiset<Ngram> ngrams = NgramEnumerator.gatherNgram(index,ids,template,minDf);
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
        BufferedWriter bufferedWriter = new BufferedWriter(new FileWriter(new File(config.getString("archive.folder"),"allFeatures.txt")));
        for (Multiset.Entry<Ngram> ngramEntry: allNgrams.entrySet()){
            bufferedWriter.write(ngramEntry.getElement().toString());
            bufferedWriter.write("\t");
            bufferedWriter.write(""+ngramEntry.getCount());
            bufferedWriter.newLine();
        }

        bufferedWriter.close();
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

    static MultiLabelClfDataSet loadTrainData(Config config, MultiLabelIndex index, FeatureList features,
                                    IdTranslator idTranslator, LabelTranslator labelTranslator) throws Exception{
        System.out.println("creating training set");
        int totalDim = features.size();
        System.out.println("allocating "+totalDim+" columns for training set");
        MultiLabelClfDataSet dataSet = loadData(config,index,features,idTranslator,totalDim,labelTranslator);
        System.out.println("training set created");
        return dataSet;
    }

    static MultiLabelClfDataSet loadTestData(Config config, MultiLabelIndex index,
                                             FeatureList features, IdTranslator idTranslator,
                                   LabelTranslator labelTranslator) throws Exception{
        System.out.println("creating test set");

        int totalDim = features.size();

        MultiLabelClfDataSet dataSet = loadData(config,index,features,idTranslator,totalDim,labelTranslator);
        System.out.println("test set created");
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

    static LabelTranslator loadTestLabelTranslator(Config config, MultiLabelIndex index, String[] testIndexIds, LabelTranslator trainLabelTranslator){
        List<String> extLabels = new ArrayList<>();
        for (int i=0;i<trainLabelTranslator.getNumClasses();i++){
            extLabels.add(trainLabelTranslator.toExtLabel(i));
        }

        Collection<Terms.Bucket> buckets = index.termAggregation(config.getString("index.labelField"), testIndexIds);
        List<String> newLabels = new ArrayList<>();
        System.out.println("label distribution in test set:");
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
            System.out.println("WARNING: found new labels in test set: "+newLabels);
        }
        return new LabelTranslator(extLabels);
    }



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
                    bw.write(index.getStringField(extId,field));
                    if (i!=fields.length-1){
                        bw.write(",");
                    }

                }
                bw.write("\n");
            }
        }

    }

    static void build(Config config, MultiLabelIndex index) throws Exception{
        int numDocsInIndex = index.getNumDocs();
        String[] trainIndexIds = sampleTrain(config,index);
        System.out.println("number of training documents = "+trainIndexIds.length);
        IdTranslator trainIdTranslator = loadIdTranslator(trainIndexIds);
        FeatureList featureList = new FeatureList();

        LabelTranslator trainLabelTranslator = loadTrainLabelTranslator(config, index, trainIndexIds);
        if (config.getBoolean("useInitialFeatures")){
            addInitialFeatures(config,index,featureList,trainIndexIds);
        }

        Set<Ngram> ngrams = gather(config,index,trainIndexIds);
        addNgramFeatures(featureList,ngrams);

        MultiLabelClfDataSet trainDataSet = loadTrainData(config,index,featureList, trainIdTranslator, trainLabelTranslator);
        System.out.println("in training set :");
//        showDistribution(config,trainDataSet,labelTranslator);


        trainDataSet.setFeatureList(featureList);
        saveDataSet(config, trainDataSet, config.getString("archive.trainingSet"));
        if (config.getBoolean("archive.dumpFields")){
            dumpTrainFields(config, index, trainIdTranslator);
        }

        String[] testIndexIds = sampleTest(numDocsInIndex,trainIndexIds);
        IdTranslator testIdTranslator = loadIdTranslator(testIndexIds);
        LabelTranslator testLabelTranslator = loadTestLabelTranslator(config, index, testIndexIds,trainLabelTranslator);

        MultiLabelClfDataSet testDataSet = loadTestData(config,index,featureList,testIdTranslator,testLabelTranslator);
        testDataSet.setFeatureList(featureList);
        saveDataSet(config, testDataSet, config.getString("archive.testSet"));
        if (config.getBoolean("archive.dumpFields")){
            dumpTestFields(config, index, testIdTranslator);
        }
    }
}
