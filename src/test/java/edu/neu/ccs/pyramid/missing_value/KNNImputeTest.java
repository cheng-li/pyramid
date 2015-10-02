package edu.neu.ccs.pyramid.missing_value;

import com.fasterxml.jackson.databind.ObjectMapper;
import edu.neu.ccs.pyramid.classification.PredictionAnalysis;
import edu.neu.ccs.pyramid.classification.boosting.lktb.*;
import edu.neu.ccs.pyramid.classification.lkboost.LKTBConfig;
import edu.neu.ccs.pyramid.classification.lkboost.LKTBInspector;
import edu.neu.ccs.pyramid.classification.lkboost.LKTBTrainer;
import edu.neu.ccs.pyramid.classification.lkboost.LKTreeBoost;
import edu.neu.ccs.pyramid.configuration.Config;
import edu.neu.ccs.pyramid.dataset.*;
import edu.neu.ccs.pyramid.eval.*;
import edu.neu.ccs.pyramid.regression.ClassScoreCalculation;
import org.apache.commons.lang3.time.StopWatch;

import java.io.*;
import java.util.ArrayList;
import java.util.List;
import java.util.regex.Pattern;

public class KNNImputeTest {
    private static final Config config = new Config("config/local.config");
    private static final String DATASETS = config.getString("input.datasets");
    private static final String TMP = config.getString("output.tmp");

    public static void main(String[] args) throws Exception {
        double accuracy_knn = census_knn();
        double accuracy_lktb = census_lktb();
        System.out.println("KNNImpute accuracy: " + accuracy_knn);
        System.out.println("LKTreeBoost accuracy: " + accuracy_lktb);
    }

    public static double census_knn() throws Exception{
        KNNImpute knnImpute = new KNNImpute();
        loadArffDataSet(new File(DATASETS, "census/train.arff"), new File(TMP, "census/train"));
        loadArffDataSet(new File(DATASETS, "census/test.arff"), new File(TMP, "census/test"));
        //generate trec files
        ClfDataSet train = StandardFormat.loadClfDataSet(2, new File(TMP, "census/train/data.txt"),
                new File(TMP, "census/train/labels.txt"), ",", DataSetType.CLF_DENSE, true);
        List<String> featureNames = loadFeatures();
        train = knnImpute.saveData(train, featureNames, 1);
        TRECFormat.save(train, new File(TMP, "census/norm_trec_data/train.trec"));
        ClfDataSet test = StandardFormat.loadClfDataSet(2, new File(TMP, "census/test/data.txt"),
                new File(TMP, "census/test/labels.txt"), ",", DataSetType.CLF_DENSE, true);
        test = knnImpute.saveData(test, featureNames, 1);
        TRECFormat.save(test, new File(TMP, "census/norm_trec_data/test.trec"));
        //impute
        ClfDataSet completeTrain = TRECFormat.loadClfDataSet(new File(DATASETS, "census/norm_trec_data/train.trec"),
                DataSetType.CLF_DENSE, true);
        ClfDataSet completeTest = TRECFormat.loadClfDataSet(new File(DATASETS, "census/norm_trec_data/test.trec"),
                DataSetType.CLF_DENSE, true);
        train = knnImpute.impute(train, completeTrain, train);
        test = knnImpute.impute(test, completeTest, train);
        TRECFormat.save(train, new File(TMP, "/census/impute_data/train.trec"));
        TRECFormat.save(test, new File(TMP, "/census/impute_data/test.trec"));
        //lktb
        census_build(new File(DATASETS, "/census/impute_data/train.trec"));
        return census_load(new File(DATASETS, "/census/impute_data/test.trec"));
    }

    public static double census_lktb() throws Exception {
        KNNImpute knnImpute = new KNNImpute();
        //generate trec files
        ClfDataSet train = StandardFormat.loadClfDataSet(2, new File(TMP, "census/train/data.txt"),
                new File(TMP, "census/train/labels.txt"), ",", DataSetType.CLF_DENSE, true);
        List<String> featureNames = loadFeatures();
        train = knnImpute.saveData(train, featureNames, 0);
        TRECFormat.save(train, new File(TMP, "census/trec_data/train.trec"));
        ClfDataSet test = StandardFormat.loadClfDataSet(2, new File(TMP, "census/test/data.txt"),
                new File(TMP, "census/test/labels.txt"), ",", DataSetType.CLF_DENSE, true);
        test = knnImpute.saveData(test, featureNames, 0);
        TRECFormat.save(test, new File(TMP, "census/trec_data/test.trec"));
        //lktb
        census_build(new File(DATASETS, "/census/trec_data/train.trec"));
        return census_load(new File(DATASETS, "/census/trec_data/test.trec"));
    }

    /*
     * Read census data
     */
    public static void loadArffDataSet(File readFile, File writeFile) {
        BufferedReader reader = null;
        BufferedWriter writer0;
        BufferedWriter writer1;
        BufferedWriter writer2;
        try {
            if (!writeFile.exists()){
                writeFile.mkdirs();
            }
            reader = new BufferedReader(new FileReader(readFile));
            writer0 = new BufferedWriter(new FileWriter(writeFile + "/feature_names.txt"));
            writer1 = new BufferedWriter(new FileWriter(writeFile + "/data.txt"));
            writer2 = new BufferedWriter(new FileWriter(writeFile + "/labels.txt"));
            String line = "";
            while (!(line = reader.readLine()).contains("@attribute")) {
            }
            int feature_num = 1;
            while (line.contains("@attribute")) {
                String[] words = line.split(" ");
                if (words[2].equals("numeric")) {
                    writer0.write(words[1] + ":" + "\t" + "continuous." + "\n");
                }
                feature_num++;
                line = reader.readLine();
            }
            writer0.close();
            while (!(line = reader.readLine()).equals("@data")) {
            }
            while ((line = reader.readLine()) != null) {
                String[] words = line.split(",");
                for (int j = 0; j < words.length-2; j++) {
                    if (words[j].equals("?")) {
                        writer1.write("NaN" + ",");
                    }
                    else {
                        writer1.write(words[j] + ",");
                    }
                }
                if (words[words.length - 2].equals("?")) {
                    writer1.write("NaN" + "," + "\n");
                }
                else {
                    writer1.write(words[words.length - 2] + "\n");
                }
                if (words[words.length - 1].equals("<=50K")) {
                    writer2.write("0\n");
                }
                else if (words[words.length - 1].equals(">50K")){
                    writer2.write("1\n");
                }
            }
            reader.close();
            writer1.close();
            writer2.close();
        } catch (IOException e) {
            e.printStackTrace();
        } finally {
            if (reader != null) {
                try {
                    reader.close();
                } catch (IOException e1) {
                }
            }
        }
    }

    static List<String> loadFeatures() throws IOException {
        List<String> names = new ArrayList<String>();
        try(BufferedReader br = new BufferedReader(new FileReader(new File(TMP, "census/train/feature_names.txt")))
        ){
            String line;
            while((line = br.readLine())!=null){
                String name = line.split(Pattern.quote(":"))[0];
                names.add(name);
            }
        }
        return names;
    }

    static double census_load(File trecFile) throws Exception{
        System.out.println("loading ensemble");
        LKTreeBoost lkTreeBoost = LKTreeBoost.deserialize(new File(TMP,"/LKTreeBoostTest/ensemble.ser"));
        ClfDataSet dataSet;
        dataSet = TRECFormat.loadClfDataSet(trecFile, DataSetType.CLF_DENSE, true);
        System.out.println("test data:");
        System.out.println(dataSet.getMetaInfo());
        System.out.println(dataSet.getLabelTranslator());


        double accuracy = Accuracy.accuracy(lkTreeBoost, dataSet);
        System.out.println(accuracy);
        System.out.println("auc = "+ AUC.auc(lkTreeBoost, dataSet));
        ConfusionMatrix confusionMatrix = new ConfusionMatrix(lkTreeBoost,dataSet);
        System.out.println("confusion matrix:");
        System.out.println(confusionMatrix.printWithExtLabels());
        System.out.println("top featureList for class 0");
        System.out.println(LKTBInspector.topFeatures(lkTreeBoost, 0));

        System.out.println(new PerClassMeasures(confusionMatrix,0));
        System.out.println(new PerClassMeasures(confusionMatrix,1));
        System.out.println("macor-averaged:");
        System.out.println(new MacroAveragedMeasures(confusionMatrix));
//        System.out.println(lkTreeBoost);


        int[] labels = dataSet.getLabels();
        List<double[]> classProbs = lkTreeBoost.predictClassProbs(dataSet);
        for (int k=0;k<dataSet.getNumClasses();k++){
            int numMatches = 0;
            double sumProbs = 0;
            for (int i=0;i<dataSet.getNumDataPoints();i++){
                if (labels[i]==k){
                    numMatches += 1;
                }
                sumProbs += classProbs.get(i)[k];
            }
            System.out.println("for class "+k);
            System.out.println("number of matches ="+numMatches);
            System.out.println("sum of probs = "+sumProbs);
        }

        LabelTranslator labelTranslator = dataSet.getLabelTranslator();
        ClassScoreCalculation classScoreCalculation = LKTBInspector.decisionProcess(lkTreeBoost,labelTranslator,dataSet.getRow(0),0,10);
        ObjectMapper mapper = new ObjectMapper();
        mapper.writeValue(new File(TMP,"score_calculation.json"), classScoreCalculation);
        PredictionAnalysis predictionAnalysis = LKTBInspector.analyzePrediction(lkTreeBoost,dataSet,0,10);
        ObjectMapper mapper1 = new ObjectMapper();
        mapper1.writeValue(new File(TMP,"prediction_analysis.json"), predictionAnalysis);

        return accuracy;
    }

    static void census_build(File trecFile) throws Exception{

        ClfDataSet dataSet;
        dataSet = TRECFormat.loadClfDataSet(trecFile, DataSetType.CLF_SPARSE, true);
        System.out.println(dataSet.getMetaInfo());

        LKTreeBoost lkTreeBoost = new LKTreeBoost(2);


        LKTBConfig trainConfig = new LKTBConfig.Builder(dataSet)
                .numLeaves(7).learningRate(0.1).numSplitIntervals(50).minDataPerLeaf(1)
                .dataSamplingRate(1).featureSamplingRate(1)
                .randomLevel(1)
                .build();

        LKTBTrainer trainer = new LKTBTrainer(trainConfig,lkTreeBoost);

        StopWatch stopWatch = new StopWatch();
        stopWatch.start();
        for (int round =0;round<500;round++){
            System.out.println("round="+round);
            trainer.iterate();
        }
        stopWatch.stop();
        System.out.println(stopWatch);


        double accuracy = Accuracy.accuracy(lkTreeBoost,dataSet);
        System.out.println("accuracy="+accuracy);

        int[] labels = dataSet.getLabels();
        List<double[]> classProbs = lkTreeBoost.predictClassProbs(dataSet);
        for (int k=0;k<dataSet.getNumClasses();k++){
            int numMatches = 0;
            double sumProbs = 0;
            for (int i=0;i<dataSet.getNumDataPoints();i++){
                if (labels[i]==k){
                    numMatches += 1;
                }
                sumProbs += classProbs.get(i)[k];
            }
            System.out.println("for class "+k);
            System.out.println("number of matches ="+numMatches);
            System.out.println("sum of probs = "+sumProbs);
        }



        lkTreeBoost.serialize(new File(TMP, "/LKTreeBoostTest/ensemble.ser"));
    }

}