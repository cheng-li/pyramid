//package edu.neu.ccs.pyramid.classification;
//
//import edu.neu.ccs.pyramid.classification.lkboost.LKTBConfig;
//import edu.neu.ccs.pyramid.classification.lkboost.LKTBTrainer;
//import edu.neu.ccs.pyramid.classification.lkboost.LKTreeBoost;
//import edu.neu.ccs.pyramid.classification.logistic_regression.ElasticNetLogisticTrainer;
//import edu.neu.ccs.pyramid.classification.logistic_regression.LogisticRegression;
//import edu.neu.ccs.pyramid.configuration.Config;
//import edu.neu.ccs.pyramid.dataset.ClfDataSet;
//import edu.neu.ccs.pyramid.dataset.DataSetType;
//import edu.neu.ccs.pyramid.dataset.TRECFormat;
//import edu.neu.ccs.pyramid.eval.Accuracy;
//import org.apache.commons.lang3.time.StopWatch;
//
//import java.io.File;
//import java.util.Arrays;
//
//public class PlattScalingTest {
//    private static final Config config = new Config("config/local.properties");
//    private static final String DATASETS = config.getString("input.datasets");
//    private static final String TMP = config.getString("output.tmp");
//
//    public static void main(String[] args) throws Exception{
////        test1();
//        test2();
//    }
//
//    private static void test1() throws Exception{
//        ClfDataSet dataSet = TRECFormat.loadClfDataSet(new File(DATASETS, "/spam/trec_data/train.trec"),
//                DataSetType.CLF_SPARSE, true);
//        System.out.println(dataSet.getMetaInfo());
//
//        LKTreeBoost lkTreeBoost = new LKTreeBoost(2);
//
//
//        LKTBConfig trainConfig = new LKTBConfig.Builder(dataSet)
//                .numLeaves(7).learningRate(0.1).numSplitIntervals(50).minDataPerLeaf(1)
//                .dataSamplingRate(1).featureSamplingRate(1)
//                .randomLevel(10)
//                .build();
//
//        LKTBTrainer trainer = new LKTBTrainer(trainConfig,lkTreeBoost);
//
//        StopWatch stopWatch = new StopWatch();
//        stopWatch.start();
//        for (int round =0;round<200;round++){
//            System.out.println("round="+round);
//            trainer.iterate();
//        }
//        stopWatch.stop();
//        System.out.println(stopWatch);
//
//
//        double accuracy = Accuracy.accuracy(lkTreeBoost, dataSet);
//        System.out.println("accuracy="+accuracy);
//
//        PlattScaling plattScaling = new PlattScaling(dataSet,lkTreeBoost);
//        for (int i=0;i<4000;i++){
//            System.out.println(Arrays.toString(lkTreeBoost.predictClassScores(dataSet.getRow(i))));
//            System.out.println(Arrays.toString(lkTreeBoost.predictClassProbs(dataSet.getRow(i))));
//            System.out.println(Arrays.toString(plattScaling.predictClassProbs(dataSet.getRow(i))));
//            System.out.println("======================");
//        }
//    }
//
//
//    private static void test2() throws Exception{
//        ClfDataSet dataSet = TRECFormat.loadClfDataSet(new File(DATASETS, "/spam/trec_data/train.trec"),
//                DataSetType.CLF_SPARSE, true);
//        System.out.println(dataSet.getMetaInfo());
//
////        RidgeLogisticTrainer trainer = RidgeLogisticTrainer.getBuilder().setGaussianPriorVariance(1)
////                .build();
////        LogisticRegression logisticRegression = trainer.train(dataSet);
//
//        LogisticRegression logisticRegression = new LogisticRegression(dataSet.getNumClasses(),dataSet.getNumFeatures());
//        ElasticNetLogisticTrainer trainer = ElasticNetLogisticTrainer.newBuilder(logisticRegression,dataSet)
//                .setRegularization(0.000000001).setL1Ratio(0).build();
//        trainer.train();
//
//        double accuracy = Accuracy.accuracy(logisticRegression, dataSet);
//        System.out.println("accuracy="+accuracy);
//
//
//        PlattScaling plattScaling = new PlattScaling(dataSet,logisticRegression);
//        for (int i=0;i<4000;i++){
//            System.out.println(Arrays.toString(logisticRegression.predictClassScores(dataSet.getRow(i))));
//            System.out.println(Arrays.toString(logisticRegression.predictClassProbs(dataSet.getRow(i))));
//            System.out.println(Arrays.toString(plattScaling.predictClassProbs(dataSet.getRow(i))));
//            System.out.println("======================");
//        }
//    }
//}