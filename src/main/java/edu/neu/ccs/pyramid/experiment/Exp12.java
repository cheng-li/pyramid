package edu.neu.ccs.pyramid.experiment;

import edu.neu.ccs.pyramid.configuration.Config;
import edu.neu.ccs.pyramid.dataset.*;
import edu.neu.ccs.pyramid.elasticsearch.ESIndex;
import edu.neu.ccs.pyramid.elasticsearch.MultiLabelIndex;
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
 * for multi label dataset,
 * dump feature matrix with initial features and unigram features
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

    static void addInitialFeatures(Config config, MultiLabelIndex index,
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
                    // missing value is not a category
                    if (!category.equals(ESIndex.STRING_MISSING_VALUE)){
                        builder.addCategory(category);
                    }
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

    static List<String> gatherUnigrams(Config config, MultiLabelIndex index,
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


    //todo keep track of feature types(numerical /binary)
    static MultiLabelClfDataSet loadData(Config config, MultiLabelIndex index,
                               FeatureMappers featureMappers,
                               IdTranslator idTranslator, int totalDim,
                               LabelTranslator labelTranslator) throws Exception{
        int numDataPoints = idTranslator.numData();
        int numClasses = labelTranslator.getNumClasses();
        MultiLabelClfDataSet dataSet;
        if(config.getBoolean("featureMatrix.sparse")){
            dataSet= new SparseMLClfDataSet(numDataPoints,totalDim,numClasses);
        } else {
            dataSet= new DenseMLClfDataSet(numDataPoints,totalDim,numClasses);
        }
        for(int i=0;i<numDataPoints;i++){
            String dataIndexId = idTranslator.toExtId(i);
            List<String> extMultiLabel = index.getExtMultiLabel(dataIndexId);
            for (String extLabel: extMultiLabel){
                int intLabel = labelTranslator.toIntLabel(extLabel);
                dataSet.addLabel(i,intLabel);
            }
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
                            // if a value is missing, set nan
                            if (category.equals(ESIndex.STRING_MISSING_VALUE)){
                                for (int featureIndex=categoricalFeatureMapper.getStart();featureIndex<categoricalFeatureMapper.getEnd();featureIndex++){
                                    dataSet.setFeatureValue(algorithmId,featureIndex,Double.NaN);
                                }
                            }
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
                            // if it is missing, it is nan automatically
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

    static MultiLabelClfDataSet loadTrainData(Config config, MultiLabelIndex index, FeatureMappers featureMappers,
                                    IdTranslator idTranslator, LabelTranslator labelTranslator) throws Exception{
        System.out.println("creating training set");
        int totalDim = featureMappers.getTotalDim();
        System.out.println("allocating "+totalDim+" columns for training set");
        MultiLabelClfDataSet dataSet = loadData(config,index,featureMappers,idTranslator,totalDim,labelTranslator);
        System.out.println("training set created");
        return dataSet;
    }

    static MultiLabelClfDataSet loadTestData(Config config, MultiLabelIndex index,
                                   FeatureMappers featureMappers, IdTranslator idTranslator,
                                   LabelTranslator labelTranslator) throws Exception{
        System.out.println("creating test set");

        int totalDim = featureMappers.getTotalDim();

        MultiLabelClfDataSet dataSet = loadData(config,index,featureMappers,idTranslator,totalDim,labelTranslator);
        System.out.println("test set created");
        return dataSet;
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

        DataSetUtil.dumpDataSettings(dataSet,new File(dataFile,"data_settings.txt"));
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
        FeatureMappers featureMappers = new FeatureMappers();

        LabelTranslator trainLabelTranslator = loadTrainLabelTranslator(index, trainIndexIds);
        if (config.getBoolean("useInitialFeatures")){
            addInitialFeatures(config,index,featureMappers,trainIndexIds);
        }

        List<String> unigrams = gatherUnigrams(config,index,trainIndexIds);
        addUnigramFeatures(featureMappers,unigrams);


        MultiLabelClfDataSet trainDataSet = loadTrainData(config,index,featureMappers, trainIdTranslator, trainLabelTranslator);
        System.out.println("in training set :");
//        showDistribution(config,trainDataSet,labelTranslator);


        DataSetUtil.setFeatureMappers(trainDataSet,featureMappers);
        saveDataSet(config, trainDataSet, config.getString("archive.trainingSet"));
        if (config.getBoolean("archive.dumpFields")){
            dumpTrainFields(config, index, trainIdTranslator);
        }

        String[] testIndexIds = sampleTest(numDocsInIndex,trainIndexIds);
        IdTranslator testIdTranslator = loadIdTranslator(testIndexIds);
        LabelTranslator testLabelTranslator = loadTestLabelTranslator(index, testIndexIds,trainLabelTranslator);

        MultiLabelClfDataSet testDataSet = loadTestData(config,index,featureMappers,testIdTranslator,testLabelTranslator);
        DataSetUtil.setFeatureMappers(testDataSet,featureMappers);
        saveDataSet(config, testDataSet, config.getString("archive.testSet"));
        if (config.getBoolean("archive.dumpFields")){
            dumpTestFields(config, index, testIdTranslator);
        }
    }
}
