package edu.neu.ccs.pyramid.classification.boosting.lktb;

import edu.neu.ccs.pyramid.dataset.ClfDataSet;
import edu.neu.ccs.pyramid.dataset.DenseClfDataSet;
import edu.neu.ccs.pyramid.dataset.SparseClfDataSet;
import edu.neu.ccs.pyramid.eval.Accuracy;
import edu.neu.ccs.pyramid.eval.ConfusionMatrix;
import org.apache.commons.lang3.time.StopWatch;

import java.io.File;
import java.util.Arrays;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

import static org.junit.Assert.*;

public class LKTreeBoostTest {
    public static void main(String[] args) throws Exception {
        spam_test();
//        newsgroup_test();
//        spam_build();
//        spam_load();
//        spam_resume_train();
//        spam_polluted_build();
//        spam_polluted_load();
//        spam_fake_build();

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
//            prediction[i] = lkTreeBoost.predict(dataSet.getFeatureRow(i));
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
//            prediction[i] = lkTreeBoost.predict(dataSet.getFeatureRow(i));
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
        LKTreeBoost lkTreeBoost = LKTreeBoost.deserialize(new File("/Users/chengli/tmp/LKTreeBoostTest/ensemble.ser"));
        File featureFile = new File("/Users/chengli/Datasets/spam/test_data.txt");
        File labelFile = new File("/Users/chengli/Datasets/spam/test_label.txt");
        ClfDataSet dataSet = DenseClfDataSet.loadStandard(featureFile, labelFile, ",");

        int numFeatures = dataSet.getNumFeatures();
        int numDataPoints = dataSet.getNumDataPoints();
        int [] labels = dataSet.getLabels();


        int[] prediction = new int[numDataPoints];
        for (int i=0;i<numDataPoints;i++){
            prediction[i] = lkTreeBoost.predict(dataSet.getFeatureRow(i));
        }
        double accuracy = Accuracy.accuracy(labels, prediction);
        System.out.println(accuracy);
        ConfusionMatrix confusionMatrix = new ConfusionMatrix(2,lkTreeBoost,dataSet);
        System.out.println("confusion matrix:");
        System.out.println(confusionMatrix);

    }

    static void spam_build() throws Exception{

        File featureFile = new File("/Users/chengli/Datasets/spam/train_data.txt");
        File labelFile = new File("/Users/chengli/Datasets/spam/train_label.txt");
        ClfDataSet dataSet = DenseClfDataSet.loadStandard(featureFile, labelFile, ",");
//        ClfDataSet dataSet = SparseClfDataSet.loadStandard(featureFile, labelFile, ",");




        LKTreeBoost lkTreeBoost = new LKTreeBoost(2);
        LKTBConfig trainConfig = new LKTBConfig.Builder(dataSet,2)
                .numLeaves(7).learningRate(0.1).
                        dataSamplingRate(1).featureSamplingRate(1).build();
        lkTreeBoost.setTrainConfig(trainConfig);

        StopWatch stopWatch = new StopWatch();
        stopWatch.start();
        for (int round =0;round<100;round++){
            System.out.println("round="+round);
            lkTreeBoost.boostOneRound();
        }
        stopWatch.stop();
        System.out.println(stopWatch);


        double accuracy = Accuracy.accuracy(lkTreeBoost,dataSet);
        System.out.println(accuracy);


        LKTreeBoost.serialize(lkTreeBoost,new File("/Users/chengli/tmp/LKTreeBoostTest/ensemble.ser"));
    }

    static void spam_resume_train_1() throws Exception{
        ExecutorService executor = Executors.newFixedThreadPool(4);

        File featureFile = new File("/Users/chengli/Datasets/spam/train_data.txt");
        File labelFile = new File("/Users/chengli/Datasets/spam/train_label.txt");
        ClfDataSet dataSet = DenseClfDataSet.loadStandard(featureFile, labelFile, ",");




        LKTreeBoost lkTreeBoost = new LKTreeBoost(2);
        LKTBConfig trainConfig = new LKTBConfig.Builder(dataSet,2)
                .numLeaves(7).learningRate(0.1).
                        dataSamplingRate(1).featureSamplingRate(1).build();
        lkTreeBoost.setTrainConfig(trainConfig);

        StopWatch stopWatch = new StopWatch();
        stopWatch.start();
        for (int round =0;round<50;round++){
            System.out.println("round="+round);
            lkTreeBoost.boostOneRound();
        }
        stopWatch.stop();
        System.out.println(stopWatch);


        double accuracy = Accuracy.accuracy(lkTreeBoost,dataSet);
        System.out.println(accuracy);

        executor.shutdown();
        LKTreeBoost.serialize(lkTreeBoost,new File("/Users/chengli/tmp/LKTreeBoostTest/ensemble.ser"));

    }

    static void spam_resume_train_2() throws Exception{
        System.out.println("loading ensemble");
        LKTreeBoost lkTreeBoost = LKTreeBoost.deserialize(new File("/Users/chengli/tmp/LKTreeBoostTest/ensemble.ser"));
        ExecutorService executor = Executors.newFixedThreadPool(4);

        File featureFile = new File("/Users/chengli/Datasets/spam/train_data.txt");
        File labelFile = new File("/Users/chengli/Datasets/spam/train_label.txt");
        ClfDataSet dataSet = DenseClfDataSet.loadStandard(featureFile, labelFile, ",");

        LKTBConfig trainConfig = new LKTBConfig.Builder(dataSet,2)
                .numLeaves(7).learningRate(0.1).
                        dataSamplingRate(1).featureSamplingRate(1).build();
        lkTreeBoost.setTrainConfig(trainConfig);

        StopWatch stopWatch = new StopWatch();
        stopWatch.start();
        for (int round =50;round<100;round++){
            System.out.println("round="+round);
            lkTreeBoost.boostOneRound();
        }
        stopWatch.stop();
        System.out.println(stopWatch);


        double accuracy = Accuracy.accuracy(lkTreeBoost,dataSet);
        System.out.println(accuracy);
        System.out.println(lkTreeBoost.getTrees(0).size());
        executor.shutdown();





    }

    static void spam_polluted_load() throws Exception{
        System.out.println("loading ensemble");
        LKTreeBoost lkTreeBoost = LKTreeBoost.deserialize(new File("/Users/chengli/tmp/LKTreeBoostTest/ensemble.ser"));
        File featureFile = new File("/Users/chengli/Datasets/spam/polluted/test_feature.txt");
        File labelFile = new File("/Users/chengli/Datasets/spam/polluted/test_label.txt");
        ClfDataSet dataSet = DenseClfDataSet.loadStandard(featureFile, labelFile, " ");

        int numFeatures = dataSet.getNumFeatures();
        int numDataPoints = dataSet.getNumDataPoints();
        int [] labels = dataSet.getLabels();


        int[] prediction = new int[numDataPoints];
        for (int i=0;i<numDataPoints;i++){
            prediction[i] = lkTreeBoost.predict(dataSet.getFeatureRow(i));
        }
        double accuracy = Accuracy.accuracy(labels, prediction);
        System.out.println(accuracy);
//        TRECDataSet.save(dataSet,new File("/Users/chengli/tmp/test.trec"));
    }

    static void spam_polluted_build() throws Exception{
        ExecutorService executor = Executors.newFixedThreadPool(4);

        File featureFile = new File("/Users/chengli/Datasets/spam/polluted/train_feature.txt");
        File labelFile = new File("/Users/chengli/Datasets/spam/polluted/train_label.txt");
        ClfDataSet dataSet = DenseClfDataSet.loadStandard(featureFile, labelFile, " ");




        LKTreeBoost lkTreeBoost = new LKTreeBoost(2);
        LKTBConfig trainConfig = new LKTBConfig.Builder(dataSet,2)
                .numLeaves(7).learningRate(0.1).
                        dataSamplingRate(1).featureSamplingRate(1).build();
        lkTreeBoost.setTrainConfig(trainConfig);

        StopWatch stopWatch = new StopWatch();
        stopWatch.start();
        for (int round =0;round<1;round++){
            System.out.println("round="+round);
            lkTreeBoost.boostOneRound();
        }
        stopWatch.stop();
        System.out.println(stopWatch);


        double accuracy = Accuracy.accuracy(lkTreeBoost,dataSet);
        System.out.println(accuracy);

        executor.shutdown();
        LKTreeBoost.serialize(lkTreeBoost,new File("/Users/chengli/tmp/LKTreeBoostTest/ensemble.ser"));
//        TRECDataSet.save(dataSet,new File("/Users/chengli/tmp/train.trec"));
    }


    static void spam_fake_build() throws Exception{
        double ratio=0.001;
        ExecutorService executor = Executors.newFixedThreadPool(4);

        File featureFile = new File("/Users/chengli/Datasets/spam/train_data.txt");
        File labelFile = new File("/Users/chengli/Datasets/spam/train_label.txt");
//        ClfDataSet dataSet = DenseClfDataSet.loadStandard(featureFile, labelFile, ",");
        ClfDataSet dataSet = DenseClfDataSet.loadStandard(featureFile, labelFile, ",");
        for (int i=0;i<dataSet.getNumDataPoints();i++){
            boolean set = Math.random()<ratio;
            if (set){
                dataSet.setLabel(i,1);
            }
        }




        LKTreeBoost lkTreeBoost = new LKTreeBoost(2);
        LKTBConfig trainConfig = new LKTBConfig.Builder(dataSet,2)
                .numLeaves(7).learningRate(0.1).
                        dataSamplingRate(1).featureSamplingRate(1).build();
        lkTreeBoost.setTrainConfig(trainConfig);

        StopWatch stopWatch = new StopWatch();
        stopWatch.start();
        for (int round =0;round<10000;round++){
            System.out.println("round="+round);
            lkTreeBoost.boostOneRound();
        }
        stopWatch.stop();
        System.out.println(stopWatch);


        double accuracy = Accuracy.accuracy(lkTreeBoost,dataSet);
        System.out.println(accuracy);

        executor.shutdown();
//        LKTreeBoost.serialize(lkTreeBoost,new File("/Users/chengli/tmp/LKTreeBoostTest/ensemble.ser"));
    }

}