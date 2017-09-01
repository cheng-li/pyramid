package edu.neu.ccs.pyramid.classification.lkboost;

import com.fasterxml.jackson.databind.ObjectMapper;
import edu.neu.ccs.pyramid.classification.PredictionAnalysis;
import edu.neu.ccs.pyramid.classification.logistic_regression.ElasticNetLogisticTrainer;
import edu.neu.ccs.pyramid.classification.logistic_regression.LogisticRegression;
import edu.neu.ccs.pyramid.classification.logistic_regression.LogisticRegressionInspector;
import edu.neu.ccs.pyramid.configuration.Config;
import edu.neu.ccs.pyramid.dataset.*;
import edu.neu.ccs.pyramid.eval.*;
import edu.neu.ccs.pyramid.regression.ClassScoreCalculation;
import edu.neu.ccs.pyramid.util.Sampling;
import org.apache.commons.lang3.time.StopWatch;

import java.io.File;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.stream.Collectors;

public class LKTreeBoostTest {
    private static final Config config = new Config("config/local.properties");
    private static final String DATASETS = config.getString("input.datasets");
    private static final String TMP = config.getString("output.tmp");
    
    public static void main(String[] args) throws Exception {
//        weightTest();
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
        LKBoost lkBoost = LKBoost.deserialize(new File(TMP, "/LKTreeBoostTest/ensemble.ser"));
        ClfDataSet dataSet = TRECFormat.loadClfDataSet(new File(DATASETS,"/spam/trec_data/test.trec"),
                DataSetType.CLF_DENSE,true);
        System.out.println("test data:");
        System.out.println(dataSet.getMetaInfo());
        System.out.println(dataSet.getLabelTranslator());


        double accuracy = Accuracy.accuracy(lkBoost, dataSet);
        System.out.println(accuracy);
        System.out.println("auc = "+ AUC.auc(lkBoost,dataSet));
        ConfusionMatrix confusionMatrix = new ConfusionMatrix(lkBoost,dataSet);
        System.out.println("confusion matrix:");
        System.out.println(confusionMatrix.printWithExtLabels());
        System.out.println("top featureList for class 0");
        System.out.println(LKBInspector.topFeatures(lkBoost, 0));

        System.out.println(new PerClassMeasures(confusionMatrix,0));
        System.out.println(new PerClassMeasures(confusionMatrix,1));
        System.out.println("macor-averaged:");
        System.out.println(new MacroAveragedMeasures(confusionMatrix));
//        System.out.println(lkTreeBoost);


        int[] labels = dataSet.getLabels();
        List<double[]> classProbs = lkBoost.predictClassProbs(dataSet);
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
        ClassScoreCalculation classScoreCalculation = LKBInspector.decisionProcess(lkBoost, labelTranslator, dataSet.getRow(0), 0, 10);
        ObjectMapper mapper = new ObjectMapper();
        mapper.writeValue(new File(TMP,"score_calculation.json"), classScoreCalculation);
        PredictionAnalysis predictionAnalysis = LKBInspector.analyzePrediction(lkBoost, dataSet, 0, 10);
        ObjectMapper mapper1 = new ObjectMapper();
        mapper1.writeValue(new File(TMP,"prediction_analysis.json"), predictionAnalysis);
    }

    static void spam_build() throws Exception{


        ClfDataSet dataSet = TRECFormat.loadClfDataSet(new File(DATASETS,"/spam/trec_data/train.trec"),
                DataSetType.CLF_SPARSE,true);
        System.out.println(dataSet.getMetaInfo());

        LKBoost lkBoost = new LKBoost(2);





        LKBoostOptimizer trainer = new LKBoostOptimizer(lkBoost,dataSet);
        trainer.initialize();

        StopWatch stopWatch = new StopWatch();
        stopWatch.start();

        for (int round =0;round<200;round++){
            System.out.println("round="+round);
            trainer.iterate();
        }
        stopWatch.stop();
        System.out.println(stopWatch);


        double accuracy = Accuracy.accuracy(lkBoost,dataSet);
        System.out.println("accuracy="+accuracy);

        int[] labels = dataSet.getLabels();
        List<double[]> classProbs = lkBoost.predictClassProbs(dataSet);
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



        lkBoost.serialize(new File(TMP, "/LKTreeBoostTest/ensemble.ser"));
    }

    /**
     * test resume training
     * first stage
     * @throws Exception
     */
    static void spam_resume_train_1() throws Exception{

        ClfDataSet dataSet = TRECFormat.loadClfDataSet(new File(DATASETS,"spam/trec_data/train.trec"),
                DataSetType.CLF_DENSE,true);

        LKBoost lkBoost = new LKBoost(2);

        LKBoostOptimizer trainer = new LKBoostOptimizer(lkBoost,dataSet);
        trainer.initialize();

        StopWatch stopWatch = new StopWatch();
        stopWatch.start();
        for (int round =0;round<50;round++){
            System.out.println("round="+round);
            trainer.iterate();
        }
        stopWatch.stop();
        System.out.println(stopWatch);


        double accuracy = Accuracy.accuracy(lkBoost,dataSet);
        System.out.println(accuracy);
        lkBoost.serialize(new File(TMP,"/LKTreeBoostTest/ensemble.ser"));

    }

    /**
     * second stage
     * @throws Exception
     */
    static void spam_resume_train_2() throws Exception{
        System.out.println("loading ensemble");
        LKBoost lkBoost = LKBoost.deserialize(new File(TMP, "/LKTreeBoostTest/ensemble.ser"));

        ClfDataSet dataSet = TRECFormat.loadClfDataSet(new File(DATASETS,"spam/trec_data/train.trec"),
                DataSetType.CLF_DENSE,true);

        LKBoostOptimizer trainer = new LKBoostOptimizer(lkBoost,dataSet);
        trainer.initialize();

        StopWatch stopWatch = new StopWatch();
        stopWatch.start();
        for (int round =50;round<100;round++){
            System.out.println("round="+round);
            trainer.iterate();
        }
        stopWatch.stop();
        System.out.println(stopWatch);


        double accuracy = Accuracy.accuracy(lkBoost,dataSet);
        System.out.println(accuracy);

    }

    /**
     * test lktb's performance on feature selection
     * @throws Exception
     */
    static void spam_polluted_load() throws Exception{
        System.out.println("loading ensemble");
        LKBoost lkBoost = LKBoost.deserialize(new File(TMP, "/LKTreeBoostTest/ensemble.ser"));
        File featureFile = new File(DATASETS,"/spam/polluted/test_feature.txt");
        File labelFile = new File(DATASETS,"spam/polluted/test_label.txt");
        ClfDataSet dataSet = StandardFormat.loadClfDataSet(2,featureFile, labelFile, " ", DataSetType.CLF_DENSE,false);


        double accuracy = Accuracy.accuracy(lkBoost, dataSet);
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

        LKBoost lkBoost = new LKBoost(2);
        LKBoostOptimizer trainer = new LKBoostOptimizer(lkBoost,dataSet);
        trainer.initialize();

        StopWatch stopWatch = new StopWatch();
        stopWatch.start();
        for (int round =0;round<100;round++){
            System.out.println("round="+round);
            trainer.iterate();
        }
        stopWatch.stop();
        System.out.println(stopWatch);


        double accuracy = Accuracy.accuracy(lkBoost,dataSet);
        System.out.println(accuracy);

        lkBoost.serialize(new File(TMP,"/LKTreeBoostTest/ensemble.ser"));
//        TRECDataSet.save(dataSet,new File("/Users/chengli/tmp/train.trec"));
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
        LKBoost lkBoost = LKBoost.deserialize(new File(TMP, "/LKTreeBoostTest/ensemble.ser"));
        ClfDataSet dataSet = TRECFormat.loadClfDataSet(new File(DATASETS,"/spam/missing_value/0.5_missing/test.trec"),
                DataSetType.CLF_DENSE,true);

        System.out.println("test data:");
        System.out.println(dataSet.getMetaInfo());


        double accuracy = Accuracy.accuracy(lkBoost, dataSet);
        System.out.println(accuracy);
        System.out.println("auc = "+ AUC.auc(lkBoost,dataSet));
        ConfusionMatrix confusionMatrix = new ConfusionMatrix(lkBoost,dataSet);
        System.out.println("confusion matrix:");
        System.out.println(confusionMatrix.printWithExtLabels());
        System.out.println("top featureList for class 0");
        System.out.println(LKBInspector.topFeatures(lkBoost, 0));

        System.out.println(new PerClassMeasures(confusionMatrix,0));
        System.out.println(new PerClassMeasures(confusionMatrix,1));
        System.out.println("macor-averaged:");
        System.out.println(new MacroAveragedMeasures(confusionMatrix));
//        System.out.println(lkTreeBoost);


        int[] labels = dataSet.getLabels();
        List<double[]> classProbs = lkBoost.predictClassProbs(dataSet);
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

        LKBoost lkBoost = new LKBoost(2);


        LKBoostOptimizer trainer = new LKBoostOptimizer(lkBoost,dataSet);
        trainer.initialize();

        StopWatch stopWatch = new StopWatch();
        stopWatch.start();
        for (int round =0;round<200;round++){
            System.out.println("round="+round);
            trainer.iterate();
        }
        stopWatch.stop();
        System.out.println(stopWatch);


        double accuracy = Accuracy.accuracy(lkBoost,dataSet);
        System.out.println("accuracy="+accuracy);

        int[] labels = dataSet.getLabels();
        List<double[]> classProbs = lkBoost.predictClassProbs(dataSet);
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



        lkBoost.serialize(new File(TMP, "/LKTreeBoostTest/ensemble.ser"));
    }

    private static void mnist_all() throws Exception{
        mnist_harr_build();
        mnist_harr_load();
    }

    static void mnist_harr_load() throws Exception{
        System.out.println("loading ensemble");
        LKBoost lkBoost = LKBoost.deserialize(new File(TMP, "/LKTreeBoostTest/ensemble.ser"));

        ClfDataSet dataSet = TRECFormat.loadClfDataSet(new File(DATASETS,"mnist/trec/test.trec"),DataSetType.CLF_DENSE,true);
        System.out.println(dataSet.getMetaInfo());

        double accuracy = Accuracy.accuracy(lkBoost, dataSet);
        System.out.println("accuracy="+accuracy);
        ConfusionMatrix confusionMatrix = new ConfusionMatrix(lkBoost,dataSet);
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
        LKBoost lkBoost = new LKBoost(10);
        LKBoostOptimizer trainer = new LKBoostOptimizer(lkBoost,dataSet);
        trainer.initialize();

        StopWatch stopWatch = new StopWatch();
        stopWatch.start();
        for (int round =0;round<100;round++){
            System.out.println("round="+round);
            trainer.iterate();
        }
        stopWatch.stop();
        System.out.println(stopWatch);


        double accuracy = Accuracy.accuracy(lkBoost,dataSet);
        System.out.println(accuracy);

        lkBoost.serialize(new File(TMP,"/LKTreeBoostTest/ensemble.ser"));
//        TRECDataSet.save(dataSet,new File("/Users/chengli/tmp/train.trec"));
    }


    static void classic3_all() throws Exception{
        classic_train();
        classic3_test();
    }
    static void classic3_test() throws Exception{
        System.out.println("loading ensemble");
        LKBoost lkBoost = LKBoost.deserialize(new File(TMP, "/LKTreeBoostTest/ensemble.ser"));

        ClfDataSet dataSet = TRECFormat.loadClfDataSet(new File(DATASETS,"classic3/classic3_exp11/test.trec"),DataSetType.CLF_DENSE,true);
        System.out.println(dataSet.getMetaInfo());

        double accuracy = Accuracy.accuracy(lkBoost, dataSet);
        System.out.println("accuracy="+accuracy);
        ConfusionMatrix confusionMatrix = new ConfusionMatrix(lkBoost,dataSet);
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
        LKBoost lkBoost = new LKBoost(3);
        LKBoostOptimizer trainer = new LKBoostOptimizer(lkBoost,dataSet);
        trainer.initialize();

        StopWatch stopWatch = new StopWatch();
        stopWatch.start();
        for (int round =0;round<100;round++){
            System.out.println("round="+round);
            trainer.iterate();
        }
        stopWatch.stop();
        System.out.println(stopWatch);


        double accuracy = Accuracy.accuracy(lkBoost,dataSet);
        System.out.println(accuracy);

        lkBoost.serialize(new File(TMP,"/LKTreeBoostTest/ensemble.ser"));
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
        LKBoost lkBoost = LKBoost.deserialize(new File(TMP, "/LKTreeBoostTest/ensemble.ser"));

        System.out.println("--------------------------------");
        ClfDataSet dataSet1 = TRECFormat.loadClfDataSet(new File(DATASETS, "faculty/train.trec"), DataSetType.CLF_DENSE, false);
        System.out.println(dataSet1.getMetaInfo());
        double accuracy = Accuracy.accuracy(lkBoost, dataSet1);
        double auc = AUC.auc(lkBoost, dataSet1);
        double precision = Precision.precision(lkBoost, dataSet1,1);
        double recall = Recall.recall(lkBoost, dataSet1, 0);
        double f1 = FMeasure.f1(precision,recall);
        System.out.println("auc="+auc);
        System.out.println("accuracy="+accuracy);
        System.out.println("precision="+precision);
        System.out.println("recall="+recall);
        System.out.println("f1="+f1);

        System.out.println("--------------------------------");
        ClfDataSet dataSet = TRECFormat.loadClfDataSet(new File(DATASETS,"faculty/test.trec"),DataSetType.CLF_DENSE,false);

        System.out.println(dataSet.getMetaInfo());
        accuracy = Accuracy.accuracy(lkBoost, dataSet);
        auc = AUC.auc(lkBoost, dataSet);
        precision = Precision.precision(lkBoost, dataSet,1);
        recall = Recall.recall(lkBoost, dataSet, 1);
        f1 = FMeasure.f1(precision,recall);
        System.out.println("auc="+auc);
        System.out.println("accuracy="+accuracy);
        System.out.println("precision="+precision);
        System.out.println("recall="+recall);
        System.out.println("f1="+f1);
        ConfusionMatrix confusionMatrix = new ConfusionMatrix(lkBoost,dataSet);
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
        LKBoost lkBoost = new LKBoost(2);
        LKBoostOptimizer trainer = new LKBoostOptimizer(lkBoost,dataSet);
        trainer.initialize();

        StopWatch stopWatch = new StopWatch();
        stopWatch.start();
        for (int round =0;round<100;round++){
            System.out.println("round="+round);
            trainer.iterate();
        }
        stopWatch.stop();
        System.out.println(stopWatch);


        double accuracy = Accuracy.accuracy(lkBoost,dataSet);
        System.out.println(accuracy);

        lkBoost.serialize(new File(TMP,"/LKTreeBoostTest/ensemble.ser"));

    }

    static void bingyu_test() throws Exception{
        ArrayList<String> arrayList = new ArrayList<>();
        arrayList.add("Non-Influence");
        arrayList.add("Influence");
        LabelTranslator labelTranslator = new LabelTranslator(arrayList);

        System.out.println("loading ensemble");
        LKBoost lkBoost = LKBoost.deserialize(new File(TMP, "/LKTreeBoostTest/ensemble.ser"));

        System.out.println("--------------------------------");
        ClfDataSet dataSet1 = TRECFormat.loadClfDataSet(new File(DATASETS, "bingyu/train.trec"), DataSetType.CLF_DENSE, false);
        System.out.println(dataSet1.getMetaInfo());
        double accuracy = Accuracy.accuracy(lkBoost, dataSet1);
        double auc = AUC.auc(lkBoost, dataSet1);
        double precision = Precision.precision(lkBoost, dataSet1,1);
        double recall = Recall.recall(lkBoost, dataSet1, 0);
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
        accuracy = Accuracy.accuracy(lkBoost, dataSet);
        auc = AUC.auc(lkBoost, dataSet);
        precision = Precision.precision(lkBoost, dataSet,1);
        recall = Recall.recall(lkBoost, dataSet, 1);
        f1 = FMeasure.f1(precision,recall);
        System.out.println("auc="+auc);
        System.out.println("accuracy="+accuracy);
        System.out.println("precision="+precision);
        System.out.println("recall="+recall);
        System.out.println("f1="+f1);
        ConfusionMatrix confusionMatrix = new ConfusionMatrix(lkBoost,dataSet);
        System.out.println(confusionMatrix.printWithIntLabels());

//        List<Feature> topFeatureIndex = LKTBInspector.topFeatures(lkTreeBoost, 0);
//        System.out.println("Top non-influence featureList index:"+topFeatureIndex);
//        List<String> topFeatureNames = LKTBInspector.topFeatureNames(lkTreeBoost,0);
//        System.out.println("Top influence featureList name:"+topFeatureNames);

        int[] predicts = lkBoost.predict(dataSet);
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
        LKBoost lkBoost = new LKBoost(2);
        LKBoostOptimizer trainer = new LKBoostOptimizer(lkBoost,dataSet);
        trainer.initialize();

        StopWatch stopWatch = new StopWatch();
        stopWatch.start();
        for (int round =0;round<100;round++){
            System.out.println("round="+round);
            trainer.iterate();
        }
        stopWatch.stop();
        System.out.println(stopWatch);


        double accuracy = Accuracy.accuracy(lkBoost,dataSet);
        System.out.println(accuracy);

        lkBoost.serialize(new File(TMP,"/LKTreeBoostTest/ensemble.ser"));
    }

    static void logisticTest() throws Exception{
        ClfDataSet dataSet = TRECFormat.loadClfDataSet(new File(DATASETS,"/spam/trec_data/train.trec"),
                DataSetType.CLF_SPARSE,true);
        System.out.println(dataSet.getMetaInfo());

        ClfDataSet testSet = TRECFormat.loadClfDataSet(new File(DATASETS,"/spam/trec_data/test.trec"),
                DataSetType.CLF_DENSE,true);

        LKBoost lkBoost = new LKBoost(2);


        LKBoostOptimizer trainer = new LKBoostOptimizer(lkBoost,dataSet);
        trainer.initialize();

        LogisticRegression logisticRegression = new LogisticRegression(dataSet.getNumClasses(),dataSet.getNumFeatures());
        ElasticNetLogisticTrainer logisticTrainer = ElasticNetLogisticTrainer.newBuilder(logisticRegression,dataSet)
                .setEpsilon(0.01).setL1Ratio(0.9).setRegularization(0.001).build();
        logisticTrainer.optimize();
        System.out.println("logistic regression accuracy = "+Accuracy.accuracy(logisticRegression,testSet));

        System.out.println("num feature used = "+ LogisticRegressionInspector.numOfUsedFeaturesCombined(logisticRegression));


//        lktbTrainer.addLogisticRegression(logisticRegression);
        System.out.println("boosting accuracy = "+Accuracy.accuracy(lkBoost,testSet));
        for (int i=0;i<100;i++){
            trainer.iterate();
            System.out.println("iteration "+i);
            System.out.println("boosting accuracy = "+Accuracy.accuracy(lkBoost,testSet));
        }

    }


    static void softTreeTest() throws Exception{
        ClfDataSet dataSet = TRECFormat.loadClfDataSet(new File(DATASETS,"/spam/trec_data/train.trec"),
                DataSetType.CLF_SPARSE,true);
        System.out.println(dataSet.getMetaInfo());

        ClfDataSet testSet = TRECFormat.loadClfDataSet(new File(DATASETS,"/spam/trec_data/test.trec"),
                DataSetType.CLF_DENSE,true);

        LKBoost lkBoost = new LKBoost(2);


        LKBoostOptimizer trainer = new LKBoostOptimizer(lkBoost,dataSet);
        trainer.initialize();
//        lktbTrainer.addLogisticRegression(logisticRegression);
        System.out.println("boosting accuracy = "+Accuracy.accuracy(lkBoost,testSet));
        for (int i=0;i<100;i++){
            trainer.iterate();
            System.out.println("iteration "+i);
            System.out.println("boosting accuracy = "+Accuracy.accuracy(lkBoost,testSet));
        }
    }

    static void weightTest() throws Exception{
        ClfDataSet trainSet = TRECFormat.loadClfDataSet(new File(DATASETS,"/spam/trec_data/train.trec"),
                DataSetType.CLF_SPARSE,true);
        ClfDataSet testSet = TRECFormat.loadClfDataSet(new File(DATASETS,"/spam/trec_data/test.trec"),
                DataSetType.CLF_SPARSE,true);
        System.out.println(trainSet.getMetaInfo());

        int[] selected = Sampling.sampleByPercentage(trainSet.getNumDataPoints(),0.5);
        double[] weights = new double[trainSet.getNumDataPoints()];
        for (int i: selected){
            weights[i]=1.0;
        }

        List<Integer> list = Arrays.stream(selected).mapToObj(i->i).collect(Collectors.toList());

        ClfDataSet subset = DataSetUtil.sampleData(trainSet, list);

        LKBoost lkBoost = new LKBoost(2);

        LKBoostOptimizer trainer = new LKBoostOptimizer(lkBoost,trainSet,weights);
        trainer.initialize();
        for (int i=0;i<100;i++){
            trainer.iterate();
            System.out.println("iteration "+i);
            System.out.println("training accuracy = "+Accuracy.accuracy(lkBoost,trainSet));
            System.out.println("test accuracy = "+Accuracy.accuracy(lkBoost,testSet));
        }


        LKBoost lkBoost2 = new LKBoost(2);

        LKBoostOptimizer trainer2 = new LKBoostOptimizer(lkBoost2,subset);
        trainer2.initialize();

        for (int i=0;i<100;i++){
            trainer2.iterate();
            System.out.println("iteration "+i);
            System.out.println("training accuracy = "+Accuracy.accuracy(lkBoost2,trainSet));
            System.out.println("test accuracy = "+Accuracy.accuracy(lkBoost2,testSet));
        }
    }

}