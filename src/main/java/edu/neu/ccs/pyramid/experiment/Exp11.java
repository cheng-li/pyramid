package edu.neu.ccs.pyramid.experiment;

import edu.neu.ccs.pyramid.configuration.Config;
import edu.neu.ccs.pyramid.dataset.*;
import edu.neu.ccs.pyramid.elasticsearch.ESIndex;
import edu.neu.ccs.pyramid.elasticsearch.ESIndexBuilder;
import edu.neu.ccs.pyramid.elasticsearch.TermStat;
import edu.neu.ccs.pyramid.feature.*;
import edu.neu.ccs.pyramid.util.Sampling;
import org.elasticsearch.action.search.SearchResponse;
import org.elasticsearch.search.SearchHit;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.util.*;
import java.util.regex.Pattern;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

/**
 * for single label dataset,
 * dump feature matrix with initial features and unigram features
 * follow exp3
 * Created by chengli on 10/10/14.
 */
public class Exp11 {
    public static void main(String[] args) throws Exception{
        if (args.length !=1){
            throw new IllegalArgumentException("please specify the config file");
        }

        Config config = new Config(args[0]);
        System.out.println(config);

        ESIndex index = loadIndex(config);
        build(config,index);
        index.close();
    }

    static ESIndex loadIndex(Config config) throws Exception{
        ESIndexBuilder builder = ESIndexBuilder.builder()
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
        ESIndex index = builder.build();
        System.out.println("index loaded");
        System.out.println("there are "+index.getNumDocs()+" documents in the index.");
//        for (int i=0;i<index.getNumDocs();i++){
//            System.out.println(i);
//            System.out.println(index.getLabel(""+i));
//        }
        return index;
    }

    static String[] sampleTrain(Config config, ESIndex index){
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
            double trainPercentage = config.getDouble("split.random.trainPercentage");
            int[] labels = new int[numDocsInIndex];
            for (int i=0;i<labels.length;i++){
                labels[i] = index.getLabel(""+i);
            }
            List<Integer> sample = Sampling.stratified(labels, trainPercentage);
            trainIds = new String[sample.size()];
            for (int i=0;i<trainIds.length;i++){
                trainIds[i] = ""+sample.get(i);
            }
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

    static void addInitialFeatures(Config config, ESIndex index,
                                   FeatureMappers featureMappers,
                                   String[] ids) throws Exception{
        String featureFieldPrefix = config.getString("index.featureFieldPrefix");

        Set<String> allFields = index.listAllFields();
        List<String> featureFields = allFields.stream().
                filter(field -> field.startsWith(featureFieldPrefix)).
                collect(Collectors.toList());
        System.out.println("all possible initial features:"+featureFields);

        for (String field: featureFields){
            String featureType = index.getFieldType(field);
            if (featureType.equalsIgnoreCase("string")){
                CategoricalFeatureMapperBuilder builder = new CategoricalFeatureMapperBuilder();
                builder.setFeatureName(field);
                builder.setStart(featureMappers.nextAvailable());
                builder.setSource("field");
                for (String id: ids){
                    String category = index.getStringField(id, field);
                    builder.addCategory(category);
                }
                boolean toAdd = true;
                CategoricalFeatureMapper mapper = builder.build();
                if (config.getBoolean("categFeature.filter")){
                    double threshold = config.getDouble("categFeature.percentThreshold");
                    int numCategories = mapper.getNumCategories();
                    if (numCategories> ids.length*threshold){
                        toAdd=false;
                        System.out.println("field "+field+" has too many categories "
                                +"("+numCategories+"), omitted.");
                    }
                }
                if(toAdd){
                    featureMappers.addMapper(mapper);
                }

            } else {
                NumericalFeatureMapperBuilder builder = new NumericalFeatureMapperBuilder();
                builder.setFeatureName(field);
                builder.setFeatureIndex(featureMappers.nextAvailable());
                builder.setSource("field");
                NumericalFeatureMapper mapper = builder.build();
                featureMappers.addMapper(mapper);
            }
        }
    }

    static List<String> gatherUnigrams(Config config, ESIndex index,
                                   String[] ids) throws Exception{
        int minDf = config.getInt("minDf");
        Set<String> unigrams = new HashSet<>();
        for (String id: ids) {
            Set<TermStat> termStats = index.getTermStats(id);
            termStats.stream().filter(termStat -> termStat.getDf() > minDf).forEach(termStat -> unigrams.add(termStat.getTerm()));
        }
        return unigrams.stream().sorted().collect(Collectors.toList());
    }

    static void addUnigramFeatures(FeatureMappers featureMappers,List<String> unigrams){
        for (String unigram: unigrams){
            int featureIndex = featureMappers.nextAvailable();
            NumericalFeatureMapper mapper = NumericalFeatureMapper.getBuilder().
                    setFeatureIndex(featureIndex).setFeatureName(unigram).
                    setSource("matching_score").build();
            featureMappers.addMapper(mapper);
        }
    }

    static ClfDataSet loadData(Config config, ESIndex index,
                               FeatureMappers featureMappers,
                               IdTranslator idTranslator, int totalDim,
                               LabelTranslator labelTranslator) throws Exception{
        int numDataPoints = idTranslator.numData();
        int numClasses = config.getInt("numClasses");
        ClfDataSet dataSet;
        if(config.getBoolean("featureMatrix.sparse")){
            dataSet= new SparseClfDataSet(numDataPoints,totalDim,numClasses);
        } else {
            dataSet= new DenseClfDataSet(numDataPoints,totalDim,numClasses);
        }
        for(int i=0;i<numDataPoints;i++){
            String dataIndexId = idTranslator.toExtId(i);
            int label = index.getLabel(dataIndexId);
            dataSet.setLabel(i,label);
        }

        String[] dataIndexIds = idTranslator.getAllExtIds();

        featureMappers.getCategoricalFeatureMappers().stream().parallel().
                forEach(categoricalFeatureMapper -> {
                    String featureName = categoricalFeatureMapper.getFeatureName();
                    String source = categoricalFeatureMapper.getSource();
                    if (source.equalsIgnoreCase("field")){
                        for (String id: dataIndexIds){
                            int algorithmId = idTranslator.toIntId(id);
                            String category = index.getStringField(id,featureName);
                            // might be a new category unseen in training
                            if (categoricalFeatureMapper.hasCategory(category)){
                                int featureIndex = categoricalFeatureMapper.getFeatureIndex(category);
                                dataSet.setFeatureValue(algorithmId,featureIndex,1);
                            }
                        }
                    }
                });


        featureMappers.getNumericalFeatureMappers().stream().parallel().
                forEach(numericalFeatureMapper -> {
                    String featureName = numericalFeatureMapper.getFeatureName();
                    String source = numericalFeatureMapper.getSource();
                    int featureIndex = numericalFeatureMapper.getFeatureIndex();

                    if (source.equalsIgnoreCase("field")){
                        for (String id: dataIndexIds){
                            int algorithmId = idTranslator.toIntId(id);
                            float value = index.getFloatField(id,featureName);
                            dataSet.setFeatureValue(algorithmId,featureIndex,value);
                        }
                    }

                    if (source.equalsIgnoreCase("matching_score")){
                        SearchResponse response = null;

                        //todo assume unigram, so slop doesn't matter
                        response = index.matchPhrase(index.getBodyField(), featureName, dataIndexIds, 0);

                        SearchHit[] hits = response.getHits().getHits();
                        for (SearchHit hit: hits){
                            String indexId = hit.getId();
                            float score = hit.getScore();
                            int algorithmId = idTranslator.toIntId(indexId);
                            dataSet.setFeatureValue(algorithmId,featureIndex,score);
                        }
                    }
                });
        DataSetUtil.setIdTranslator(dataSet, idTranslator);
        DataSetUtil.setLabelTranslator(dataSet, labelTranslator);
        return dataSet;
    }

    static ClfDataSet loadTrainData(Config config, ESIndex index, FeatureMappers featureMappers,
                                    IdTranslator idTranslator, LabelTranslator labelTranslator) throws Exception{
        System.out.println("creating training set");
        int totalDim = featureMappers.getTotalDim();
        System.out.println("allocating "+totalDim+" columns for training set");
        ClfDataSet dataSet = loadData(config,index,featureMappers,idTranslator,totalDim,labelTranslator);
        System.out.println("training set created");
        return dataSet;
    }

    static ClfDataSet loadTestData(Config config, ESIndex index,
                                   FeatureMappers featureMappers, IdTranslator idTranslator,
                                   LabelTranslator labelTranslator) throws Exception{
        System.out.println("creating test set");

        int totalDim = featureMappers.getTotalDim();

        ClfDataSet dataSet = loadData(config,index,featureMappers,idTranslator,totalDim,labelTranslator);
        System.out.println("test set created");
        return dataSet;
    }

    static LabelTranslator loadLabelTranslator(Config config, ESIndex index) throws Exception{
        int numClasses = config.getInt("numClasses");
        int numDocs = index.getNumDocs();
        Map<Integer, String> map = new HashMap<>(numClasses);
        for (int i=0;i<numDocs;i++){
            int intLabel = index.getLabel(""+i);
            String extLabel = index.getExtLabel("" + i);
            map.put(intLabel,extLabel);
        }
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
        DataSetUtil.dumpDataSettings(dataSet,new File(dataFile,"data_settings.txt"));
        DataSetUtil.dumpFeatureSettings(dataSet,new File(dataFile,"feature_settings.txt"));
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
        int numDocsInIndex = index.getNumDocs();
        String[] trainIndexIds = sampleTrain(config,index);
        System.out.println("number of training documents = "+trainIndexIds.length);
        IdTranslator trainIdTranslator = loadIdTranslator(trainIndexIds);
        FeatureMappers featureMappers = new FeatureMappers();

        LabelTranslator labelTranslator = loadLabelTranslator(config, index);
        if (config.getBoolean("useInitialFeatures")){
            addInitialFeatures(config,index,featureMappers,trainIndexIds);
        }

        List<String> unigrams = gatherUnigrams(config,index,trainIndexIds);
        addUnigramFeatures(featureMappers,unigrams);


        ClfDataSet trainDataSet = loadTrainData(config,index,featureMappers, trainIdTranslator, labelTranslator);
        System.out.println("in training set :");
        showDistribution(config,trainDataSet,labelTranslator);


        DataSetUtil.setFeatureMappers(trainDataSet,featureMappers);
        saveDataSet(config, trainDataSet, config.getString("archive.trainingSet"));
        if (config.getBoolean("archive.dumpFields")){
            dumpTrainFeatures(config,index,trainIdTranslator);
        }

        String[] testIndexIds = sampleTest(numDocsInIndex,trainIndexIds);
        IdTranslator testIdTranslator = loadIdTranslator(testIndexIds);

        ClfDataSet testDataSet = loadTestData(config,index,featureMappers,testIdTranslator,labelTranslator);
        DataSetUtil.setFeatureMappers(testDataSet,featureMappers);
        saveDataSet(config, testDataSet, config.getString("archive.testSet"));
        if (config.getBoolean("archive.dumpFields")){
            dumpTestFeatures(config,index,testIdTranslator);
        }
    }


}
