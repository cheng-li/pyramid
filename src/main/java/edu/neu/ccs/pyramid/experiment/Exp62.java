package edu.neu.ccs.pyramid.experiment;

import com.google.common.collect.EvictingQueue;
import edu.neu.ccs.pyramid.classification.logistic_regression.LogisticLoss;
import edu.neu.ccs.pyramid.classification.logistic_regression.LogisticRegression;
import edu.neu.ccs.pyramid.configuration.Config;
import edu.neu.ccs.pyramid.dataset.*;
import edu.neu.ccs.pyramid.elasticsearch.ESIndex;
import edu.neu.ccs.pyramid.elasticsearch.SingleLabelIndex;
import edu.neu.ccs.pyramid.elasticsearch.TermStat;
import edu.neu.ccs.pyramid.eval.Accuracy;
import edu.neu.ccs.pyramid.feature.FeatureMappers;
import edu.neu.ccs.pyramid.feature.NumericalFeatureMapper;
import edu.neu.ccs.pyramid.feature_extraction.NgramEnumerator;
import edu.neu.ccs.pyramid.feature_extraction.UncertainSampler;
import edu.neu.ccs.pyramid.optimization.LBFGS;
import edu.neu.ccs.pyramid.util.Pair;
import edu.neu.ccs.pyramid.util.Sampling;
import org.apache.commons.io.FileUtils;
import org.apache.commons.io.IOUtils;
import org.apache.commons.lang3.time.StopWatch;
import org.elasticsearch.action.search.SearchResponse;
import org.elasticsearch.search.SearchHit;

import java.io.*;
import java.util.*;
import java.util.concurrent.ConcurrentHashMap;
import java.util.regex.Pattern;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

/**
 * extract all frequent ngrams from a document, check how probability estimation changes
 * Created by chengli on 1/26/15.
 */
public class Exp62 {
    public static void main(String[] args) throws Exception{
        if (args.length !=1){
            throw new IllegalArgumentException("please specify the config file");
        }

        Config config = new Config(args[0]);
        System.out.println(config);

        SingleLabelIndex index = loadIndex(config);


        train(config,index);


        index.close();

    }

    public static void mainFromConfig(Config config) throws Exception{

        SingleLabelIndex index = loadIndex(config);


        train(config,index);


        index.close();

    }

    static void train(Config config, SingleLabelIndex index) throws Exception{
        int totalDim = config.getInt("featureMatrix.maxNumColumns");
        LabelTranslator labelTranslator = loadLabelTranslator(config, index);
        FeatureMappers featureMappers = new FeatureMappers();
        loadInitialFeaturesFromFile(config,featureMappers);
        Set<String> duplidate = loadDuplicate(config);
        String[] trainIndexIds = sampleTrain(config,index,duplidate);
        IdTranslator trainIdTranslator = loadIdTranslator(trainIndexIds);

        ClfDataSet trainDataSet = loadData(config, index, featureMappers,trainIdTranslator,totalDim,labelTranslator);
        System.out.println("in training set :");
        showDistribution(config,trainDataSet,trainDataSet.getSetting().getLabelTranslator());

        String[] validIndexIds = sampleValid(config,index,duplidate);
        IdTranslator validIdTranslator = loadIdTranslator(validIndexIds);
        ClfDataSet validDataSet = loadData(config, index, featureMappers,validIdTranslator,totalDim,labelTranslator);
        System.out.println("in validation set :");
        showDistribution(config,validDataSet,validDataSet.getSetting().getLabelTranslator());

        String[] testIndexIds = sampleTest(config,index);
        IdTranslator testIdTranslator = loadIdTranslator(testIndexIds);
        ClfDataSet testDataSet = loadData(config, index, featureMappers,testIdTranslator,totalDim,labelTranslator);
        System.out.println("in test set :");
        showDistribution(config,testDataSet,testDataSet.getSetting().getLabelTranslator());

        trainModel(config,trainDataSet,validDataSet, testDataSet, featureMappers,
                index, trainDataSet.getSetting().getIdTranslator());

        //only keep used columns
        trainDataSet = DataSetUtil.trim(trainDataSet, featureMappers.getTotalDim());
        DataSetUtil.setFeatureMappers(trainDataSet,featureMappers);
        saveDataSet(config, trainDataSet, config.getString("archive.trainingSet"));
        if (config.getBoolean("archive.dumpFields")){
            dumpTrainFeatures(config,index,trainDataSet.getSetting().getIdTranslator());
        }


        validDataSet = DataSetUtil.trim(validDataSet, featureMappers.getTotalDim());
        DataSetUtil.setFeatureMappers(validDataSet,featureMappers);
        saveDataSet(config, validDataSet, config.getString("archive.validSet"));

        testDataSet = DataSetUtil.trim(testDataSet, featureMappers.getTotalDim());
        DataSetUtil.setFeatureMappers(testDataSet,featureMappers);
        saveDataSet(config, testDataSet, config.getString("archive.testSet"));




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




    static void trainModel(Config config, ClfDataSet trainSet, ClfDataSet validSet, ClfDataSet testSet,
                           FeatureMappers featureMappers,
                           SingleLabelIndex index, IdTranslator trainIdTranslator) throws Exception{
        String archive = config.getString("archive.folder");
        File archiveFolder = new File(archive);
        archiveFolder.mkdirs();
        int numIterations = config.getInt("train.numIterations");
        int numClasses = trainSet.getNumClasses();

        String modelName = config.getString("archive.model");

        LabelTranslator labelTranslator = trainSet.getSetting().getLabelTranslator();

        List<List<String>> longDocs = longDocs(index,validSet.getSetting().getIdTranslator().getAllExtIds(),validSet.getNumClasses());

        StopWatch stopWatch = new StopWatch();
        stopWatch.start();

        System.out.println("training model ");


        LogisticRegression logisticRegression = new LogisticRegression(numClasses,trainSet.getNumFeatures());
        logisticRegression.setFeatureExtraction(true);
        LogisticLoss logisticLoss = new LogisticLoss(logisticRegression,
                trainSet,config.getDouble("train.gaussianPriorVariance"));
        LBFGS lbfgs;

        Set<String> frequentNgrams = new HashSet<>(gather(config,index,trainIdTranslator.getAllExtIds()));
        System.out.println("number of frequent ngrams = "+frequentNgrams.size());

        Set<String> topNgrams = loadTopNgrams(config);

        Set<String> blackList = new HashSet<>();

        UncertainSampler sampler = new UncertainSampler(validSet);

        //add initial unigrams to blacklist
        for (int i=0;i<featureMappers.getTotalDim();i++){
            blackList.add(featureMappers.getName(i));
        }

        List<Integer> ns = config.getIntegers("ngram.n");

        File statsFile = new File(config.getString("archive.folder"),"stats");
        BufferedWriter statsWriter = new BufferedWriter(new FileWriter(statsFile));

        statsWriter.write("initially");
        statsWriter.write(",");
        statsWriter.write("number of features = " + featureMappers.getTotalDim());
        statsWriter.newLine();



        List<EvictingQueue<Integer>> lastSampleAlgorithmIds = new ArrayList<>();
        for (int k=0;k<numClasses;k++){
            lastSampleAlgorithmIds.add(EvictingQueue.<Integer>create(1));
        }

        List<EvictingQueue<String>> lastSampleIndexIds = new ArrayList<>();
        for (int k=0;k<numClasses;k++){
            lastSampleIndexIds.add(EvictingQueue.<String>create(1));
        }

        List<EvictingQueue<double[]>> lastSampleProbs = new ArrayList<>();
        for (int k=0;k<numClasses;k++){
            lastSampleProbs.add(EvictingQueue.<double[]>create(1));
        }

        EvictingQueue<Double> trainAccHis = EvictingQueue.create(1);
        EvictingQueue<Double> validAccHis = EvictingQueue.create(1);
        EvictingQueue<Double> testAccHis = EvictingQueue.create(1);

        List<List<String>> setCoverDocs = new ArrayList<>();
        if (!config.getString("input.setCoverDocs").equals("")){
            FileReader fileReader = new FileReader(config.getString("input.setCoverDocs"));
            List<String> lines = IOUtils.readLines(fileReader);
            for (String line: lines){
                List<String> list = new ArrayList<>();
                for (String doc: line.split(",")){
                    list.add(doc.trim());
                }
                setCoverDocs.add(list);
            }
        }

        System.out.println("set cover docs "+setCoverDocs);
        for (int iteration=0;iteration<numIterations;iteration++) {
            System.out.println("iteration " + iteration);
            System.out.println("total number of features = "+featureMappers.getTotalDim());

            logisticLoss.refresh();
            System.out.println("loss at the start of iteration " + iteration + " = " + logisticLoss.getValue());
            lbfgs = new LBFGS(logisticLoss);
            lbfgs.optimize();
            System.out.println("loss after the optimization " + iteration + " = " + logisticLoss.getValue());
            double trainAcc = Accuracy.accuracy(logisticRegression, trainSet);
            double validAcc = Accuracy.accuracy(logisticRegression, validSet);
            double testAcc = Accuracy.accuracy(logisticRegression, testSet);
            System.out.print("accuracy on training set = " + trainAcc+"\t");
            if (iteration>=1){
                System.out.print("previous accuracy on training set = " + trainAccHis.peek());
            }
            System.out.println();
            System.out.print("accuracy on validation set = " + validAcc+"\t");
            if (iteration>=1){
                System.out.print("previous accuracy on validation set = " + validAccHis.peek());
            }
            System.out.println();
            System.out.print("accuracy on test set = " + testAcc+"\t");
            if (iteration>=1){
                System.out.print("previous accuracy on test set = " + testAccHis.peek());
            }
            System.out.println();

            trainAccHis.add(trainAcc);
            validAccHis.add(validAcc);
            testAccHis.add(testAcc);


            if (iteration>=1){
                System.out.println("change of probability estimations:");
                for (int k=0;k<numClasses;k++){
                    System.out.println("for class "+k);
                    System.out.println("previous focus doc index id = "+lastSampleIndexIds.get(k).peek());
                    System.out.println("previous probability estimation = "+Arrays.toString(lastSampleProbs.get(k).peek()));
                    int algorithmId = lastSampleAlgorithmIds.get(k).peek();
                    System.out.println("new probability estimation = "
                            +Arrays.toString(logisticRegression.predictClassProbs(validSet.getRow(algorithmId))));
                }
            }



            boolean condition1 = (featureMappers.getTotalDim()
                    + config.getInt("extraction.topN")
                    < trainSet.getNumFeatures());


            boolean shouldExtractFeatures = condition1;

            if (!shouldExtractFeatures) {
                if (!condition1) {
                    System.out.println("we have reached the max number of columns " +
                            "and will not extract new features");
                    break;
                }
            }



            sampler.setClassProbMatrix(logisticRegression.predictClassProbs(validSet));


            if (shouldExtractFeatures) {

                System.out.println("start extraction ");


                for (int k = 0; k < numClasses; k++) {
                    System.out.println("for class "+k);
                    //phrases
                    int sampleAlgorithmId = 0;
                    if (config.getString("sample.fashion").equals("uncertain")){
                        sampleAlgorithmId = sampler.getUncertainOne(k).get();
                    } else if (config.getString("sample.fashion").equals("random")){
                        sampleAlgorithmId = sampler.getRandomOne(k).get();
                    } else if (config.getString("sample.fashion").equals("hard")) {
                        sampleAlgorithmId = sampler.getHardOne(k).get();
                    } else if (config.getString("sample.fashion").equals("easy")){
                            sampleAlgorithmId = sampler.getEasyOne(k).get();
                    } else if (config.getString("sample.fashion").equals("setCover")){
                        sampleAlgorithmId = validSet.getSetting().getIdTranslator().toIntId(setCoverDocs.get(k).get(iteration));
                    } else if (config.getString("sample.fashion").equals("long")) {
                        sampleAlgorithmId = validSet.getSetting().getIdTranslator().toIntId(longDocs.get(k).get(iteration));
                    } else {
                        throw new RuntimeException("illegal fashion");
                    }


                    sampler.getBlackList().add(sampleAlgorithmId);
                    String sampleIndexId = validSet.getSetting().getIdTranslator().toExtId(sampleAlgorithmId);
                    System.out.println("focus doc's index id = "+sampleIndexId);
                    System.out.println("focus doc's probability estimation = "+Arrays.toString(logisticRegression.predictClassProbs(validSet.getRow(sampleAlgorithmId))));
                    System.out.println("focus doc's true label = "+validSet.getLabels()[sampleAlgorithmId]);

                    lastSampleAlgorithmIds.get(k).add(sampleAlgorithmId);
                    lastSampleIndexIds.get(k).add(sampleIndexId);
                    lastSampleProbs.get(k).add(logisticRegression.predictClassProbs(validSet.getRow(sampleAlgorithmId)));
                    Map<Integer,String> termVector = index.getTermVector(sampleIndexId);

                    Set<String> allCandidates = new HashSet<>();
                    for (int n:ns){
                        allCandidates.addAll(NgramEnumerator.getNgramCounts(termVector, n).keySet());
                    }

                    allCandidates.retainAll(frequentNgrams);

                    if (config.getBoolean("extraction.topFeaturesOnly")){
                        allCandidates.retainAll(topNgrams);
                    }

                    allCandidates.removeAll(blackList);

                    List<String> featuresToAdd = allCandidates.stream().sorted().collect(Collectors.toList());

                    System.out.println("phrases extracted for class " + k + " (" + labelTranslator.toExtLabel(k) + "):");
                    System.out.println(featuresToAdd);

                    int startFeatureIndex = featureMappers.nextAvailable();
                    addFeaturesToMatrix(index, trainSet, featuresToAdd, startFeatureIndex);
                    addFeaturesToMatrix(index, validSet, featuresToAdd, startFeatureIndex);
                    addFeaturesToMatrix(index, testSet, featuresToAdd, startFeatureIndex);

                    addFeaturesToMappers(featureMappers,featuresToAdd);
                    blackList.addAll(featuresToAdd);
                }
                statsWriter.write("iteration = " + iteration);
                statsWriter.write(",");
                statsWriter.write("number of features = " + featureMappers.getTotalDim());
                statsWriter.write(",");
                statsWriter.write("test acc = "+ Accuracy.accuracy(logisticRegression, testSet));
                statsWriter.newLine();
            }
        }
        statsWriter.close();

        File serializedModel =  new File(archive,modelName);
        logisticRegression.serialize(serializedModel);
        System.out.println("model saved to "+serializedModel.getAbsolutePath());
        System.out.println("accuracy on training set = "+ Accuracy.accuracy(logisticRegression,
                trainSet));
        System.out.println("time spent = "+stopWatch);

    }



    static void showDistribution(Config config, ClfDataSet dataSet, LabelTranslator labelTranslator){
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


    //todo
    /**
     * assuming unigram for now
     */
    static void loadInitialFeaturesFromFile(Config config, FeatureMappers featureMappers) throws Exception{

        File initialFeatureFile = new File(config.getString("input.initialFeatureFile"));
        String[] line = FileUtils.readLines(initialFeatureFile).get(0).split(",");
        List<String> unigrams = Arrays.stream(line).collect(Collectors.toList());
        System.out.println("initial features:");
        System.out.println(unigrams);
        for (String unigram: unigrams){
            int featureIndex = featureMappers.nextAvailable();
            NumericalFeatureMapper mapper = NumericalFeatureMapper.getBuilder().
                    setFeatureIndex(featureIndex).setFeatureName(unigram).
                    setSource("matching_score").build();
            featureMappers.addMapper(mapper);
        }
    }

    static String[] sampleTrain(Config config, SingleLabelIndex index, Set<String> duplicate){
        int numDocsInIndex = index.getNumDocs();
        String[] ids = null;

        String splitField = config.getString("index.splitField");
        List<String> train = IntStream.range(0, numDocsInIndex).parallel()
                .filter(i -> index.getStringField("" + i, splitField).
                        equalsIgnoreCase("train")).
                        mapToObj(i -> "" + i).filter(id -> !duplicate.contains(id)).sorted().
                        collect(Collectors.toList());

        //todo parameter?
        ids = IntStream.range(0, train.size())
//               .filter(i-> i%5!=0)
                .mapToObj(train::get).toArray(String[]::new);
        return ids;
    }

    static String[] sampleValid(Config config, SingleLabelIndex index, Set<String> duplicate){
        int numDocsInIndex = index.getNumDocs();
        String[] ids = null;

        String splitField = config.getString("index.splitField");
        List<String> train = IntStream.range(0, numDocsInIndex).parallel()
                .filter(i -> index.getStringField("" + i, splitField).
                        equalsIgnoreCase("train")).
                        mapToObj(i -> "" + i).filter(id -> !duplicate.contains(id)).sorted().
                        collect(Collectors.toList());

        //todo: not using validation
        ids = IntStream.range(0, train.size())
//                .filter(i-> i%5==0)
                .mapToObj(train::get).toArray(String[]::new);
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

    static IdTranslator loadIdTranslator(String[] indexIds) throws Exception{
        IdTranslator idTranslator = new IdTranslator();
        for (int i=0;i<indexIds.length;i++){
            idTranslator.addData(i,""+indexIds[i]);
        }
        return idTranslator;
    }

    static ClfDataSet loadData(Config config, SingleLabelIndex index,
                               FeatureMappers featureMappers,
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

    static LabelTranslator loadLabelTranslator(Config config, SingleLabelIndex index) throws Exception{
        System.out.println("loading label translator...");
        int numClasses = config.getInt("numClasses");
        int numDocs = index.getNumDocs();
        Map<Integer, String> map = new ConcurrentHashMap<>();
        while(map.size()<numClasses){
            int i = Sampling.intUniform(0, numDocs - 1);
            int intLabel = index.getLabel(""+i);
            String extLabel = index.getExtLabel("" + i);
            map.put(intLabel,extLabel);
        }
        System.out.println("loaded");
        return new LabelTranslator(map);
    }

    static Set<String> loadDuplicate(Config config) throws Exception{
        File file = new File(config.getString("input.duplicate"));
        String[] strArr = FileUtils.readFileToString(file).split(",");
        Set<String> set = new HashSet<>();
        Arrays.stream(strArr).forEach(set::add);
        return set;
    }

    static List<String> gather(Config config, SingleLabelIndex index,
                               String[] ids) throws Exception{
        List<Integer> ns = config.getIntegers("ngram.n");
        List<Integer> minDfs = config.getIntegers("ngram.minDf");
        List<String> list = new ArrayList<>();
        for (int i=0;i<ns.size();i++){
            int n = ns.get(i);
            int minDf = minDfs.get(i);
            if (n==1){
                list.addAll(gatherUnigrams(index,ids,minDf));
            } else {
                list.addAll(gatherNgrams(index, ids, n, minDf));
            }
        }
        return list;
    }

    static List<String> gatherUnigrams(ESIndex index,
                                       String[] ids, int minDf) throws Exception{
        System.out.println("gathering unigrams with minDf "+minDf);
        Set<TermStat> unigrams = Collections.newSetFromMap(new ConcurrentHashMap<TermStat, Boolean>());
        Arrays.stream(ids).parallel().forEach(id -> {
            Set<TermStat> termStats = null;
            try {
                termStats = index.getTermStats(id);
            } catch (IOException e) {
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

    static List<String> gatherNgrams(ESIndex index,
                                     String[] ids, int n, int minDf) throws Exception{

        System.out.println("gathering "+n+"-grams with minDf "+minDf);
        List<String> ngrams = NgramEnumerator.gatherNgrams(index,ids,n,minDf);
        System.out.println("done");
        System.out.println("there are "+ngrams.size()+" "+n+"-grams");
        return ngrams;
    }

    //we have to preserve order, so use a list
    static void addFeaturesToMatrix(ESIndex index, ClfDataSet dataSet, List<String> features, int startFeatureIndex){
        IdTranslator idTranslator = dataSet.getSetting().getIdTranslator();
        List<Pair<String, SearchResponse>> searchResponseList = features.stream().parallel()
                .map(phrase -> new Pair<>(phrase, index.matchPhrase(index.getBodyField(),
                        phrase, idTranslator.getAllExtIds(), 0)))
                .collect(Collectors.toList());

        int featureIndex = startFeatureIndex;
        for (Pair<String, SearchResponse> pair : searchResponseList) {
            SearchResponse response = pair.getSecond();
            for (SearchHit hit : response.getHits().getHits()) {
                String indexId = hit.getId();
                int algorithmId = idTranslator.toIntId(indexId);
                float score = hit.getScore();
                dataSet.setFeatureValue(algorithmId, featureIndex, score);
            }

            featureIndex += 1;
        }
    }

    static void addFeaturesToMappers(FeatureMappers featureMappers, List<String> features){
        features.stream().forEach(feature ->{
            int featureIndex = featureMappers.nextAvailable();
            NumericalFeatureMapper mapper = NumericalFeatureMapper.getBuilder().
                    setFeatureIndex(featureIndex).setFeatureName(feature).
                    setSource("matching_score").build();
            featureMappers.addMapper(mapper);
        });
    }

    static List<List<String>> longDocs(SingleLabelIndex index, String[] ids, int numClasses){
        List<List<String>> all = new ArrayList<>();
        Comparator<Pair<String,Integer>> comparator = Comparator.comparing(Pair::getSecond);
        IntStream.range(0,numClasses).forEach(k-> {
            List<String> list = Arrays.stream(ids).parallel().
                    filter(id -> index.getLabel(id)==k)
                    .map(id -> new Pair<>(id, index.getDocLength(id)))
                    .sorted(comparator.reversed()).map(Pair::getFirst).collect(Collectors.toList());
            all.add(list);
        });
        return all;
    }

    static Set<String> loadTopNgrams(Config config) throws Exception{
        Set<String> set = new HashSet<>();
        File file = new File(config.getString("input.topFeatures"));
        BufferedReader bufferedReader = new BufferedReader(new FileReader(file));
        List<String> lines = IOUtils.readLines(bufferedReader);
        bufferedReader.close();
        for (String line: lines){
            for (String str: line.split(",")){
                set.add(str.trim());
            }
        }
        return set;
    }
}
