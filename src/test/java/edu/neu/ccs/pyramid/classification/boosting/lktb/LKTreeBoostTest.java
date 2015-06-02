package edu.neu.ccs.pyramid.classification.boosting.lktb;

import com.fasterxml.jackson.databind.ObjectMapper;
import edu.neu.ccs.pyramid.classification.PredictionAnalysis;
import edu.neu.ccs.pyramid.classification.logistic_regression.ElasticNetLogisticTrainer;
import edu.neu.ccs.pyramid.classification.logistic_regression.LogisticRegression;
import edu.neu.ccs.pyramid.classification.logistic_regression.LogisticRegressionInspector;
import edu.neu.ccs.pyramid.configuration.Config;
import edu.neu.ccs.pyramid.dataset.*;
import edu.neu.ccs.pyramid.eval.*;
import edu.neu.ccs.pyramid.feature.Feature;
import edu.neu.ccs.pyramid.regression.ClassScoreCalculation;
import edu.neu.ccs.pyramid.regression.regression_tree.LeafOutputType;
import org.apache.commons.lang3.time.StopWatch;

import java.io.File;
import java.util.ArrayList;
import java.util.List;

public class LKTreeBoostTest {
    private static final Config config = new Config("configs/local.config");
    private static final String DATASETS = config.getString("input.datasets");
    private static final String TMP = config.getString("output.tmp");
    
    public static void main(String[] args) throws Exception {
        spam_test();
//        newsgroup_test();
//        spam_build();
//        spam_load();
//        spam_resume_train();
//        spam_polluted_build();
//        spam_polluted_load();
//        spam_fake_build();
//        spam_missing_all();
//        mnist_all();
//        classic3_all();
//        bingyu_all();
//        faculty_all();
//        logisticTest();
//        softTreeTest();
    }

    static void spam_resume_train() throws Exception{
        spam_resume_train_1();
        spam_resume_train_2();
    }
//
//    static void newsgroup_test() throws Exception{
//        newsgroup_build();
//        newsgroup_load();
//    }
//    static void newsgroup_build() throws Exception{
//        ExecutorService executor = Executors.newFixedThreadPool(4);
//        File dataFile = new File("/Users/chengli/Datasets/20newsgroup/train.txt");
//
//        ClfDataSet dataSet = TRECDataSet.loadClfDataSet(dataFile, DataSetType.CLF_DENSE);
//        dataSet.sortAllFeatures();
//        //System.out.println(sortedDataSet);
//        int numFeatures = dataSet.getNumFeatures();
//        int numDataPoints = dataSet.getNumDataPoints();
//        int [] labels = dataSet.getLabels();
//
//        boolean[] featuresToConsider = new boolean[numFeatures];
//        Arrays.fill(featuresToConsider, true);
//
//        long startTime = System.currentTimeMillis();
//        LKTreeBoost lkTreeBoost = new LKTreeBoost(20);
//        LKTBConfig trainConfig = new LKTBConfig.Builder(executor,dataSet,20)
//                .numLeaves(2).learningRate(0.1).build();
//        lkTreeBoost.setTrainConfig(trainConfig);
//        for (int round =0;round<10;round++){
//            System.out.println("round="+round);
//            lkTreeBoost.boostOneRound();
//        }
//
//
//        int[] prediction = new int[numDataPoints];
//        for (int i=0;i<numDataPoints;i++){
//            prediction[i] = lkTreeBoost.predict(dataSet.getRow(i));
//        }
//        double accuracy = Accuracy.accuracy(labels, prediction);
//        System.out.println(accuracy);
//        long endTime   = System.currentTimeMillis();
//        double totalTime = ((double)(endTime - startTime))/1000;
//        System.out.println(totalTime);
//        executor.shutdown();
//        LKTreeBoost.serialize(lkTreeBoost,new File("/Users/chengli/tmp/LKTreeBoostTest/ensemble.ser"));
//    }
//
//    static void newsgroup_load() throws Exception{
//        System.out.println("loading ensemble");
//        LKTreeBoost lkTreeBoost = LKTreeBoost.deserialize(new File("/Users/chengli/tmp/LKTreeBoostTest/ensemble.ser"));
//        File dataFile = new File("/Users/chengli/Datasets/20newsgroup/test.txt");
//
//        ClfDataSet dataSet = TRECDataSet.loadClfDataSet(dataFile, DataSetType.CLF_SPARSE);
//
//        int numFeatures = dataSet.getNumFeatures();
//        int numDataPoints = dataSet.getNumDataPoints();
//        int [] labels = dataSet.getLabels();
//
//
//        int[] prediction = new int[numDataPoints];
//        for (int i=0;i<numDataPoints;i++){
//            prediction[i] = lkTreeBoost.predict(dataSet.getRow(i));
//        }
//        double accuracy = Accuracy.accuracy(labels, prediction);
//        System.out.println(accuracy);
//
//    }


    static void spam_test() throws Exception{
        spam_build();
        spam_load();
    }
    static void spam_load() throws Exception{
        System.out.println("loading ensemble");
        LKTreeBoost lkTreeBoost = LKTreeBoost.deserialize(new File(TMP,"/LKTreeBoostTest/ensemble.ser"));
        ClfDataSet dataSet = TRECFormat.loadClfDataSet(new File(DATASETS,"/spam/trec_data/test.trec"),
                DataSetType.CLF_DENSE,true);
        System.out.println("test data:");
        System.out.println(dataSet.getMetaInfo());
        System.out.println(dataSet.getLabelTranslator());


        double accuracy = Accuracy.accuracy(lkTreeBoost, dataSet);
        System.out.println(accuracy);
        System.out.println("auc = "+ AUC.auc(lkTreeBoost,dataSet));
        ConfusionMatrix confusionMatrix = new ConfusionMatrix(lkTreeBoost,dataSet);
        System.out.println("confusion matrix:");
        System.out.println(confusionMatrix.printWithExtLabels());
        System.out.println("top featureList for class 0");
        System.out.println(LKTBInspector.topFeatures(lkTreeBoost,0));

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
    }

    static void spam_build() throws Exception{


        ClfDataSet dataSet = TRECFormat.loadClfDataSet(new File(DATASETS,"/spam/trec_data/train.trec"),
                DataSetType.CLF_SPARSE,true);
        System.out.println(dataSet.getMetaInfo());

        LKTreeBoost lkTreeBoost = new LKTreeBoost(2);


        LKTBConfig trainConfig = new LKTBConfig.Builder(dataSet)
                .numLeaves(2).learningRate(0.1).numSplitIntervals(50).minDataPerLeaf(1)
                        .dataSamplingRate(1).featureSamplingRate(1)
                        .randomLevel(1)
                        .considerHardTree(true)
                        .considerExpectationTree(false)
                        .considerProbabilisticTree(false)
                        .setLeafOutputType(LeafOutputType.AVERAGE)
                        .build();

        LKTBTrainer trainer = new LKTBTrainer(trainConfig,lkTreeBoost);

        StopWatch stopWatch = new StopWatch();
        stopWatch.start();
        trainer.addPriorRegressors();
        for (int round =0;round<200;round++){
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

    /**
     * test resume training
     * first stage
     * @throws Exception
     */
    static void spam_resume_train_1() throws Exception{

        ClfDataSet dataSet = TRECFormat.loadClfDataSet(new File(DATASETS,"spam/trec_data/train.trec"),
                DataSetType.CLF_DENSE,true);

        LKTreeBoost lkTreeBoost = new LKTreeBoost(2);
        LKTBConfig trainConfig = new LKTBConfig.Builder(dataSet)
                .numLeaves(7).learningRate(0.1).
                        dataSamplingRate(1).featureSamplingRate(1).build();
        LKTBTrainer trainer = new LKTBTrainer(trainConfig,lkTreeBoost);

        StopWatch stopWatch = new StopWatch();
        stopWatch.start();
        for (int round =0;round<50;round++){
            System.out.println("round="+round);
            trainer.iterate();
        }
        stopWatch.stop();
        System.out.println(stopWatch);


        double accuracy = Accuracy.accuracy(lkTreeBoost,dataSet);
        System.out.println(accuracy);
        lkTreeBoost.serialize(new File(TMP,"/LKTreeBoostTest/ensemble.ser"));

    }

    /**
     * second stage
     * @throws Exception
     */
    static void spam_resume_train_2() throws Exception{
        System.out.println("loading ensemble");
        LKTreeBoost lkTreeBoost = LKTreeBoost.deserialize(new File(TMP,"/LKTreeBoostTest/ensemble.ser"));

        ClfDataSet dataSet = TRECFormat.loadClfDataSet(new File(DATASETS,"spam/trec_data/train.trec"),
                DataSetType.CLF_DENSE,true);

        LKTBConfig trainConfig = new LKTBConfig.Builder(dataSet)
                .numLeaves(7).learningRate(0.1).
                        dataSamplingRate(1).featureSamplingRate(1).build();
        LKTBTrainer trainer = new LKTBTrainer(trainConfig,lkTreeBoost);

        StopWatch stopWatch = new StopWatch();
        stopWatch.start();
        for (int round =50;round<100;round++){
            System.out.println("round="+round);
            trainer.iterate();
        }
        stopWatch.stop();
        System.out.println(stopWatch);


        double accuracy = Accuracy.accuracy(lkTreeBoost,dataSet);
        System.out.println(accuracy);
        System.out.println(lkTreeBoost.getRegressors(0).size());
    }

    /**
     * test lktb's performance on feature selection
     * @throws Exception
     */
    static void spam_polluted_load() throws Exception{
        System.out.println("loading ensemble");
        LKTreeBoost lkTreeBoost = LKTreeBoost.deserialize(new File(TMP,"/LKTreeBoostTest/ensemble.ser"));
        File featureFile = new File(DATASETS,"/spam/polluted/test_feature.txt");
        File labelFile = new File(DATASETS,"spam/polluted/test_label.txt");
        ClfDataSet dataSet = StandardFormat.loadClfDataSet(2,featureFile, labelFile, " ", DataSetType.CLF_DENSE,false);


        double accuracy = Accuracy.accuracy(lkTreeBoost, dataSet);
        System.out.println(accuracy);
//        TRECDataSet.save(dataSet,new File("/Users/chengli/tmp/test.trec"));
    }

    /**
     * test lktb's performance on feature selection
     * @throws Exception
     */
    static void spam_polluted_build() throws Exception{
        File featureFile = new File(DATASETS,"spam/polluted/train_feature.txt");
        File labelFile = new File(DATASETS,"spam/polluted/train_label.txt");
        ClfDataSet dataSet = StandardFormat.loadClfDataSet(2,featureFile, labelFile, " ", DataSetType.CLF_DENSE,false);

        LKTreeBoost lkTreeBoost = new LKTreeBoost(2);
        LKTBConfig trainConfig = new LKTBConfig.Builder(dataSet)
                .numLeaves(7).learningRate(0.1).
                        dataSamplingRate(1).featureSamplingRate(1).build();
        LKTBTrainer trainer = new LKTBTrainer(trainConfig,lkTreeBoost);

        StopWatch stopWatch = new StopWatch();
        stopWatch.start();
        for (int round =0;round<100;round++){
            System.out.println("round="+round);
            trainer.iterate();
        }
        stopWatch.stop();
        System.out.println(stopWatch);


        double accuracy = Accuracy.accuracy(lkTreeBoost,dataSet);
        System.out.println(accuracy);

        lkTreeBoost.serialize(new File(TMP,"/LKTreeBoostTest/ensemble.ser"));
//        TRECDataSet.save(dataSet,new File("/Users/chengli/tmp/train.trec"));
    }


    static void spam_fake_build() throws Exception{
        double ratio=0.001;
        File featureFile = new File(DATASETS,"/spam/train_data.txt");
        File labelFile = new File(DATASETS,"/spam/train_label.txt");
//        ClfDataSet dataSet = DenseClfDataSet.loadStandard(featureFile, labelFile, ",");
        ClfDataSet dataSet = StandardFormat.loadClfDataSet(2,featureFile, labelFile, ",", DataSetType.CLF_DENSE,false);
        for (int i=0;i<dataSet.getNumDataPoints();i++){
            boolean set = Math.random()<ratio;
            if (set){
                dataSet.setLabel(i,1);
            }
        }

        LKTreeBoost lkTreeBoost = new LKTreeBoost(2);
        LKTBConfig trainConfig = new LKTBConfig.Builder(dataSet)
                .numLeaves(7).learningRate(0.1).
                        dataSamplingRate(1).featureSamplingRate(1).build();
        LKTBTrainer trainer = new LKTBTrainer(trainConfig,lkTreeBoost);

        StopWatch stopWatch = new StopWatch();
        stopWatch.start();
        for (int round =0;round<10000;round++){
            System.out.println("round="+round);
            trainer.iterate();
        }
        stopWatch.stop();
        System.out.println(stopWatch);


        double accuracy = Accuracy.accuracy(lkTreeBoost,dataSet);
        System.out.println(accuracy);

//        LKTreeBoost.serialize(lkTreeBoost,new File("/Users/chengli/tmp/LKTreeBoostTest/ensemble.ser"));
    }


    /**
     * spam with missing values
     * @throws Exception
     */
    static void spam_missing_all() throws Exception{
        spam_missing_build();
        spam_missing_load();
    }
    static void spam_missing_load() throws Exception{
        System.out.println("loading ensemble");
        LKTreeBoost lkTreeBoost = LKTreeBoost.deserialize(new File(TMP,"/LKTreeBoostTest/ensemble.ser"));
        ClfDataSet dataSet = TRECFormat.loadClfDataSet(new File(DATASETS,"/spam/missing_value/0.5_missing/test.trec"),
                DataSetType.CLF_DENSE,true);

        System.out.println("test data:");
        System.out.println(dataSet.getMetaInfo());


        double accuracy = Accuracy.accuracy(lkTreeBoost, dataSet);
        System.out.println(accuracy);
        System.out.println("auc = "+ AUC.auc(lkTreeBoost,dataSet));
        ConfusionMatrix confusionMatrix = new ConfusionMatrix(lkTreeBoost,dataSet);
        System.out.println("confusion matrix:");
        System.out.println(confusionMatrix.printWithExtLabels());
        System.out.println("top featureList for class 0");
        System.out.println(LKTBInspector.topFeatures(lkTreeBoost,0));

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
    }

    static void spam_missing_build() throws Exception{


        ClfDataSet dataSet = TRECFormat.loadClfDataSet(new File(DATASETS,"/spam/missing_value/0.5_missing/train.trec"),
                DataSetType.CLF_DENSE,true);


        System.out.println(dataSet.getMetaInfo());

        LKTreeBoost lkTreeBoost = new LKTreeBoost(2);


        LKTBConfig trainConfig = new LKTBConfig.Builder(dataSet)
                .numLeaves(7).learningRate(0.1).numSplitIntervals(50).minDataPerLeaf(1)
                .dataSamplingRate(1).featureSamplingRate(1).build();
        LKTBTrainer trainer = new LKTBTrainer(trainConfig,lkTreeBoost);

        StopWatch stopWatch = new StopWatch();
        stopWatch.start();
        for (int round =0;round<200;round++){
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

    private static void mnist_all() throws Exception{
        mnist_harr_build();
        mnist_harr_load();
    }

    static void mnist_harr_load() throws Exception{
        System.out.println("loading ensemble");
        LKTreeBoost lkTreeBoost = LKTreeBoost.deserialize(new File(TMP,"/LKTreeBoostTest/ensemble.ser"));

        ClfDataSet dataSet = TRECFormat.loadClfDataSet(new File(DATASETS,"mnist/trec/test.trec"),DataSetType.CLF_DENSE,true);
        System.out.println(dataSet.getMetaInfo());

        double accuracy = Accuracy.accuracy(lkTreeBoost, dataSet);
        System.out.println("accuracy="+accuracy);
        ConfusionMatrix confusionMatrix = new ConfusionMatrix(lkTreeBoost,dataSet);
        System.out.println(confusionMatrix.printWithExtLabels());
        for (int k=0;k<dataSet.getNumClasses();k++){
            System.out.println("for class "+k);
            System.out.println(new PerClassMeasures(confusionMatrix,k));
        }
        System.out.println("macro-averaged:");
        System.out.println(new MacroAveragedMeasures(confusionMatrix));

    }

    /**
     * test lktb's performance on feature selection
     * @throws Exception
     */
    static void mnist_harr_build() throws Exception{

        ClfDataSet dataSet  = TRECFormat.loadClfDataSet(new File(DATASETS,"mnist/trec/train.trec"),DataSetType.CLF_DENSE,true);
        System.out.println(dataSet.getMetaInfo());
        LKTreeBoost lkTreeBoost = new LKTreeBoost(10);
        LKTBConfig trainConfig = new LKTBConfig.Builder(dataSet)
                .numLeaves(7).learningRate(0.1).
                        dataSamplingRate(1).featureSamplingRate(1).build();
        LKTBTrainer trainer = new LKTBTrainer(trainConfig,lkTreeBoost);

        StopWatch stopWatch = new StopWatch();
        stopWatch.start();
        for (int round =0;round<100;round++){
            System.out.println("round="+round);
            trainer.iterate();
        }
        stopWatch.stop();
        System.out.println(stopWatch);


        double accuracy = Accuracy.accuracy(lkTreeBoost,dataSet);
        System.out.println(accuracy);

        lkTreeBoost.serialize(new File(TMP,"/LKTreeBoostTest/ensemble.ser"));
//        TRECDataSet.save(dataSet,new File("/Users/chengli/tmp/train.trec"));
    }


    static void classic3_all() throws Exception{
        classic_train();
        classic3_test();
    }
    static void classic3_test() throws Exception{
        System.out.println("loading ensemble");
        LKTreeBoost lkTreeBoost = LKTreeBoost.deserialize(new File(TMP,"/LKTreeBoostTest/ensemble.ser"));

        ClfDataSet dataSet = TRECFormat.loadClfDataSet(new File(DATASETS,"classic3/classic3_exp11/test.trec"),DataSetType.CLF_DENSE,true);
        System.out.println(dataSet.getMetaInfo());

        double accuracy = Accuracy.accuracy(lkTreeBoost, dataSet);
        System.out.println("accuracy="+accuracy);
        ConfusionMatrix confusionMatrix = new ConfusionMatrix(lkTreeBoost,dataSet);
        System.out.println(confusionMatrix.printWithExtLabels());
        for (int k=0;k<dataSet.getNumClasses();k++){
            System.out.println("for class "+k);
            System.out.println(new PerClassMeasures(confusionMatrix,k));
        }
        System.out.println("macro-averaged:");
        System.out.println(new MacroAveragedMeasures(confusionMatrix));

    }

    /**
     * test lktb's performance on feature selection
     * @throws Exception
     */
    static void classic_train() throws Exception{

        ClfDataSet dataSet  = TRECFormat.loadClfDataSet(new File(DATASETS,"classic3/classic3_exp11/train.trec"),DataSetType.CLF_DENSE,true);
        System.out.println(dataSet.getMetaInfo());
        LKTreeBoost lkTreeBoost = new LKTreeBoost(3);
        LKTBConfig trainConfig = new LKTBConfig.Builder(dataSet)
                .numLeaves(5).learningRate(0.1).
                        dataSamplingRate(1).featureSamplingRate(1).build();
        LKTBTrainer trainer = new LKTBTrainer(trainConfig,lkTreeBoost);

        StopWatch stopWatch = new StopWatch();
        stopWatch.start();
        for (int round =0;round<100;round++){
            System.out.println("round="+round);
            trainer.iterate();
        }
        stopWatch.stop();
        System.out.println(stopWatch);


        double accuracy = Accuracy.accuracy(lkTreeBoost,dataSet);
        System.out.println(accuracy);

        lkTreeBoost.serialize(new File(TMP,"/LKTreeBoostTest/ensemble.ser"));
    }

    static void bingyu_all() throws Exception{
        bingyu_train();
        bingyu_test();
    }

    static void faculty_all() throws  Exception{
        faculty_train();
        faculty_test();
    }

    private static void faculty_test() throws Exception {
        ArrayList<String> arrayList = new ArrayList<>();

        System.out.println("loading ensemble");
        LKTreeBoost lkTreeBoost = LKTreeBoost.deserialize(new File(TMP,"/LKTreeBoostTest/ensemble.ser"));

        System.out.println("--------------------------------");
        ClfDataSet dataSet1 = TRECFormat.loadClfDataSet(new File(DATASETS, "faculty/train.trec"), DataSetType.CLF_DENSE, false);
        System.out.println(dataSet1.getMetaInfo());
        double accuracy = Accuracy.accuracy(lkTreeBoost, dataSet1);
        double auc = AUC.auc(lkTreeBoost, dataSet1);
        double precision = Precision.precision(lkTreeBoost, dataSet1,1);
        double recall = Recall.recall(lkTreeBoost, dataSet1, 0);
        double f1 = FMeasure.f1(precision,recall);
        System.out.println("auc="+auc);
        System.out.println("accuracy="+accuracy);
        System.out.println("precision="+precision);
        System.out.println("recall="+recall);
        System.out.println("f1="+f1);

        System.out.println("--------------------------------");
        ClfDataSet dataSet = TRECFormat.loadClfDataSet(new File(DATASETS,"faculty/test.trec"),DataSetType.CLF_DENSE,false);

        System.out.println(dataSet.getMetaInfo());
        accuracy = Accuracy.accuracy(lkTreeBoost, dataSet);
        auc = AUC.auc(lkTreeBoost, dataSet);
        precision = Precision.precision(lkTreeBoost, dataSet,1);
        recall = Recall.recall(lkTreeBoost, dataSet, 1);
        f1 = FMeasure.f1(precision,recall);
        System.out.println("auc="+auc);
        System.out.println("accuracy="+accuracy);
        System.out.println("precision="+precision);
        System.out.println("recall="+recall);
        System.out.println("f1="+f1);
        ConfusionMatrix confusionMatrix = new ConfusionMatrix(lkTreeBoost,dataSet);
        System.out.println(confusionMatrix.printWithIntLabels());

//        List<Feature> topFeatureIndex = LKTBInspector.topFeatures(lkTreeBoost,0);
//        System.out.println("Top non-influence featureList index:"+topFeatureIndex);

//        int[] predicts = lkTreeBoost.predict(dataSet);
//        int[] labels = dataSet.getLabels();
//        for (int i=0; i<predicts.length; i++) {
//            if (predicts[i] != labels[i]) {
//                System.out.println("DataPoint: " + i);
//                System.out.println(LKTBInspector.analyzeMistake(lkTreeBoost,dataSet.getRow(i),
//                        labels[i], predicts[i], labelTranslator, 100));
//            }
//        }
    }

    private static void faculty_train() throws Exception {

        ClfDataSet dataSet  = TRECFormat.loadClfDataSet(new File(DATASETS,"faculty/train.trec"),DataSetType.CLF_DENSE,false);
        System.out.println(dataSet.getMetaInfo());
        LKTreeBoost lkTreeBoost = new LKTreeBoost(2);
        LKTBConfig trainConfig = new LKTBConfig.Builder(dataSet)
                .numLeaves(3).learningRate(0.5).numSplitIntervals(1000)
                .dataSamplingRate(1).featureSamplingRate(1).build();
        LKTBTrainer trainer = new LKTBTrainer(trainConfig,lkTreeBoost);

        StopWatch stopWatch = new StopWatch();
        stopWatch.start();
        for (int round =0;round<100;round++){
            System.out.println("round="+round);
            trainer.iterate();
        }
        stopWatch.stop();
        System.out.println(stopWatch);


        double accuracy = Accuracy.accuracy(lkTreeBoost,dataSet);
        System.out.println(accuracy);

        lkTreeBoost.serialize(new File(TMP,"/LKTreeBoostTest/ensemble.ser"));

    }

    static void bingyu_test() throws Exception{
        ArrayList<String> arrayList = new ArrayList<>();
        arrayList.add("Non-Influence");
        arrayList.add("Influence");
        LabelTranslator labelTranslator = new LabelTranslator(arrayList);

        System.out.println("loading ensemble");
        LKTreeBoost lkTreeBoost = LKTreeBoost.deserialize(new File(TMP,"/LKTreeBoostTest/ensemble.ser"));

        System.out.println("--------------------------------");
        ClfDataSet dataSet1 = TRECFormat.loadClfDataSet(new File(DATASETS, "bingyu/train.trec"), DataSetType.CLF_DENSE, false);
        System.out.println(dataSet1.getMetaInfo());
        double accuracy = Accuracy.accuracy(lkTreeBoost, dataSet1);
        double auc = AUC.auc(lkTreeBoost, dataSet1);
        double precision = Precision.precision(lkTreeBoost, dataSet1,1);
        double recall = Recall.recall(lkTreeBoost, dataSet1, 0);
        double f1 = FMeasure.f1(precision,recall);
        System.out.println("auc="+auc);
        System.out.println("accuracy="+accuracy);
//        System.out.println("precision="+precision);
//        System.out.println("recall="+recall);
//        System.out.println("f1="+f1);

        System.out.println("--------------------------------");
        ClfDataSet dataSet = TRECFormat.loadClfDataSet(new File(DATASETS,"bingyu/test.trec"),DataSetType.CLF_DENSE,false);
        dataSet.getFeatureList().get(0).setName("dump");
        dataSet.getFeatureList().get(1).setName("CoAuthor");
        dataSet.getFeatureList().get(2).setName("CiteAuthorOut");
        dataSet.getFeatureList().get(3).setName("CiteAuthorIn");
        dataSet.getFeatureList().get(4).setName("CoAppearVenue");
        dataSet.getFeatureList().get(5).setName("PPR for CoAuthor");
        dataSet.getFeatureList().get(6).setName("PPR for CiteAuthor");

        System.out.println(dataSet.getMetaInfo());
        accuracy = Accuracy.accuracy(lkTreeBoost, dataSet);
        auc = AUC.auc(lkTreeBoost, dataSet);
        precision = Precision.precision(lkTreeBoost, dataSet,1);
        recall = Recall.recall(lkTreeBoost, dataSet, 1);
        f1 = FMeasure.f1(precision,recall);
        System.out.println("auc="+auc);
        System.out.println("accuracy="+accuracy);
        System.out.println("precision="+precision);
        System.out.println("recall="+recall);
        System.out.println("f1="+f1);
        ConfusionMatrix confusionMatrix = new ConfusionMatrix(lkTreeBoost,dataSet);
        System.out.println(confusionMatrix.printWithIntLabels());

//        List<Feature> topFeatureIndex = LKTBInspector.topFeatures(lkTreeBoost, 0);
//        System.out.println("Top non-influence featureList index:"+topFeatureIndex);
//        List<String> topFeatureNames = LKTBInspector.topFeatureNames(lkTreeBoost,0);
//        System.out.println("Top influence featureList name:"+topFeatureNames);

        int[] predicts = lkTreeBoost.predict(dataSet);
        int[] labels = dataSet.getLabels();
        for (int i=0; i<predicts.length; i++) {
            if (predicts[i] != labels[i]) {
                System.out.println("DataPoint: " + i);
//                System.out.println(LKTBInspector.analyzeMistake(lkTreeBoost,dataSet.getRow(i),
//                        labels[i], predicts[i], labelTranslator, 100));
            }
        }

    }


    static void bingyu_train() throws Exception{
        ClfDataSet dataSet  = TRECFormat.loadClfDataSet(new File(DATASETS,"bingyu/train.trec"),DataSetType.CLF_DENSE,false);
        System.out.println(dataSet.getMetaInfo());
        LKTreeBoost lkTreeBoost = new LKTreeBoost(2);
        LKTBConfig trainConfig = new LKTBConfig.Builder(dataSet)
                .numLeaves(4).learningRate(0.1).numSplitIntervals(1000)
                        .dataSamplingRate(1).featureSamplingRate(1).build();
        LKTBTrainer trainer = new LKTBTrainer(trainConfig,lkTreeBoost);

        StopWatch stopWatch = new StopWatch();
        stopWatch.start();
        for (int round =0;round<100;round++){
            System.out.println("round="+round);
            trainer.iterate();
        }
        stopWatch.stop();
        System.out.println(stopWatch);


        double accuracy = Accuracy.accuracy(lkTreeBoost,dataSet);
        System.out.println(accuracy);

        lkTreeBoost.serialize(new File(TMP,"/LKTreeBoostTest/ensemble.ser"));
    }

    static void logisticTest() throws Exception{
        ClfDataSet dataSet = TRECFormat.loadClfDataSet(new File(DATASETS,"/spam/trec_data/train.trec"),
                DataSetType.CLF_SPARSE,true);
        System.out.println(dataSet.getMetaInfo());

        ClfDataSet testSet = TRECFormat.loadClfDataSet(new File(DATASETS,"/spam/trec_data/test.trec"),
                DataSetType.CLF_DENSE,true);

        LKTreeBoost lkTreeBoost = new LKTreeBoost(2);


        LKTBConfig trainConfig = new LKTBConfig.Builder(dataSet)
                .numLeaves(7).learningRate(0.1).numSplitIntervals(50).minDataPerLeaf(1)
                .dataSamplingRate(1).featureSamplingRate(1)
                .randomLevel(1)
                .build();

        LogisticRegression logisticRegression = new LogisticRegression(dataSet.getNumClasses(),dataSet.getNumFeatures());
        ElasticNetLogisticTrainer logisticTrainer = ElasticNetLogisticTrainer.newBuilder(logisticRegression,dataSet)
                .setEpsilon(0.01).setL1Ratio(0.9).setRegularization(0.001).build();
        logisticTrainer.train();
        System.out.println("logistic regression accuracy = "+Accuracy.accuracy(logisticRegression,testSet));

        System.out.println("num feature used = "+ LogisticRegressionInspector.numOfUsedFeaturesCombined(logisticRegression));

        LKTBTrainer lktbTrainer = new LKTBTrainer(trainConfig,lkTreeBoost);
//        lktbTrainer.addLogisticRegression(logisticRegression);
        System.out.println("boosting accuracy = "+Accuracy.accuracy(lkTreeBoost,testSet));
        for (int i=0;i<100;i++){
            lktbTrainer.iterate();
            System.out.println("iteration "+i);
            System.out.println("boosting accuracy = "+Accuracy.accuracy(lkTreeBoost,testSet));
        }
        System.out.println(lkTreeBoost.getRegressors(0).get(0).predict(testSet.getRow(0)));
        System.out.println(lkTreeBoost.getRegressors(0).get(1).predict(testSet.getRow(0)));
    }


    static void softTreeTest() throws Exception{
        ClfDataSet dataSet = TRECFormat.loadClfDataSet(new File(DATASETS,"/spam/trec_data/train.trec"),
                DataSetType.CLF_SPARSE,true);
        System.out.println(dataSet.getMetaInfo());

        ClfDataSet testSet = TRECFormat.loadClfDataSet(new File(DATASETS,"/spam/trec_data/test.trec"),
                DataSetType.CLF_DENSE,true);

        LKTreeBoost lkTreeBoost = new LKTreeBoost(2);


        LKTBConfig trainConfig = new LKTBConfig.Builder(dataSet)
                .numLeaves(7).learningRate(0.1).numSplitIntervals(50).minDataPerLeaf(1)
                .dataSamplingRate(1).featureSamplingRate(1)
                .randomLevel(1)
                .build();

        LKTBTrainer lktbTrainer = new LKTBTrainer(trainConfig,lkTreeBoost);
//        lktbTrainer.addLogisticRegression(logisticRegression);
        System.out.println("boosting accuracy = "+Accuracy.accuracy(lkTreeBoost,testSet));
        for (int i=0;i<100;i++){
            lktbTrainer.iterate();
            System.out.println("iteration "+i);
            System.out.println("boosting accuracy = "+Accuracy.accuracy(lkTreeBoost,testSet));
        }
    }

}