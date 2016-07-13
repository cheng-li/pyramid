package edu.neu.ccs.pyramid.classification.logistic_regression;

import edu.neu.ccs.pyramid.configuration.Config;
import edu.neu.ccs.pyramid.dataset.ClfDataSet;
import edu.neu.ccs.pyramid.dataset.DataSetType;
import edu.neu.ccs.pyramid.dataset.TRECFormat;
import edu.neu.ccs.pyramid.eval.Accuracy;
import edu.neu.ccs.pyramid.util.Grid;
import org.apache.commons.lang3.time.StopWatch;
import org.apache.logging.log4j.Level;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.core.LoggerContext;
import org.apache.logging.log4j.core.config.Configuration;
import org.apache.logging.log4j.core.config.LoggerConfig;


import java.io.File;
import java.util.Comparator;
import java.util.List;
import java.util.stream.Collectors;

public class ElasticNetLogisticTrainerTest {
    private static final Config config = new Config("config/local.properties");
    private static final String DATASETS = config.getString("input.datasets");
    private static final String TMP = config.getString("output.tmp");
    public static void main(String[] args) throws Exception{
        LoggerContext ctx = (LoggerContext) LogManager.getContext(false);
        Configuration config = ctx.getConfiguration();
        LoggerConfig loggerConfig = config.getLoggerConfig(LogManager.ROOT_LOGGER_NAME);
        loggerConfig.setLevel(Level.OFF);
        ctx.updateLoggers();
        test1();
    }

    private static void test1() throws Exception{
        ClfDataSet dataSet = TRECFormat.loadClfDataSet(new File(DATASETS, "/spam/trec_data/train.trec"),
                DataSetType.CLF_SPARSE, true);
        ClfDataSet testSet = TRECFormat.loadClfDataSet(new File(DATASETS, "/spam/trec_data/test.trec"),
                DataSetType.CLF_SPARSE, true);
        System.out.println(dataSet.getMetaInfo());
        LogisticRegression logisticRegression = new LogisticRegression(dataSet.getNumClasses(),dataSet.getNumFeatures());
        ElasticNetLogisticTrainer trainer = ElasticNetLogisticTrainer.newBuilder(logisticRegression,dataSet)
                .setEpsilon(0.01).setL1Ratio(0.5).setRegularization(0.0001).setLineSearch(true).build();
        for (int i=0;i<100;i++){
            System.out.println("iteration "+i);
            trainer.iterate();
            System.out.println("training accuracy = "+ Accuracy.accuracy(logisticRegression,dataSet));
            System.out.println("test accuracy = "+ Accuracy.accuracy(logisticRegression,testSet));
        }

    }

    private static void test2() throws Exception{
        ClfDataSet dataSet = TRECFormat.loadClfDataSet(new File(DATASETS, "/spam/trec_data/train.trec"),
                DataSetType.CLF_SPARSE, true);
        ClfDataSet testSet = TRECFormat.loadClfDataSet(new File(DATASETS, "/spam/trec_data/test.trec"),
                DataSetType.CLF_SPARSE, true);
        System.out.println(dataSet.getMetaInfo());
        LogisticRegression logisticRegression = new LogisticRegression(dataSet.getNumClasses(),dataSet.getNumFeatures());
        ElasticNetLogisticTrainer trainer = ElasticNetLogisticTrainer.newBuilder(logisticRegression, dataSet)
                .setEpsilon(0.01).setL1Ratio(0.5).setRegularization(0.0001).build();

        trainer.optimize();
        System.out.println("training accuracy = "+ Accuracy.accuracy(logisticRegression,dataSet));
        System.out.println("test accuracy = "+ Accuracy.accuracy(logisticRegression,testSet));


    }

    private static void test3() throws Exception{
        ClfDataSet dataSet = TRECFormat.loadClfDataSet(new File(DATASETS, "/imdb/3/train.trec"),
                DataSetType.CLF_SPARSE, true);
        ClfDataSet testSet = TRECFormat.loadClfDataSet(new File(DATASETS, "/imdb/3/test.trec"),
                DataSetType.CLF_SPARSE, true);
        System.out.println(dataSet.getMetaInfo());
        LogisticRegression logisticRegression = new LogisticRegression(dataSet.getNumClasses(),dataSet.getNumFeatures());
        ElasticNetLogisticTrainer trainer = ElasticNetLogisticTrainer.newBuilder(logisticRegression, dataSet)
                .setEpsilon(0.01).setL1Ratio(1).setRegularization(0.01).build();

        for (int i=0;i<10;i++){
            System.out.println("iteration "+i);
            trainer.iterate();
            System.out.println("training accuracy = "+ Accuracy.accuracy(logisticRegression,dataSet));
            System.out.println("test accuracy = "+ Accuracy.accuracy(logisticRegression,testSet));
        }


    }

    private static void test4() throws Exception{
        ClfDataSet dataSet = TRECFormat.loadClfDataSet(new File(DATASETS, "/cnn/4/train.trec"),
                DataSetType.CLF_SPARSE, true);
        ClfDataSet testSet = TRECFormat.loadClfDataSet(new File(DATASETS, "/cnn/4/test.trec"),
                DataSetType.CLF_SPARSE, true);
        System.out.println(dataSet.getMetaInfo());
        LogisticRegression logisticRegression = new LogisticRegression(dataSet.getNumClasses(),dataSet.getNumFeatures());
        ElasticNetLogisticTrainer trainer = ElasticNetLogisticTrainer.newBuilder(logisticRegression, dataSet)
                .setEpsilon(0.01).setL1Ratio(1).setRegularization(2.4201282647943795E-4).build();

        StopWatch stopWatch = new StopWatch();
        stopWatch.start();
        trainer.optimize();
        System.out.println(stopWatch);
        System.out.println("training accuracy = "+ Accuracy.accuracy(logisticRegression,dataSet));
        System.out.println("test accuracy = "+ Accuracy.accuracy(logisticRegression,testSet));
        System.out.println("number of non-zeros= "+logisticRegression.getWeights().getAllWeights().getNumNonZeroElements());


    }

    private static void test5() throws Exception{
        ClfDataSet dataSet = TRECFormat.loadClfDataSet(new File(DATASETS, "/cnn/4/train.trec"),
                DataSetType.CLF_SPARSE, true);
        ClfDataSet testSet = TRECFormat.loadClfDataSet(new File(DATASETS, "/cnn/4/test.trec"),
                DataSetType.CLF_SPARSE, true);
        System.out.println(dataSet.getMetaInfo());
        LogisticRegression logisticRegression = new LogisticRegression(dataSet.getNumClasses(),dataSet.getNumFeatures());

        Comparator<Double> comparator = Comparator.comparing(Double::doubleValue);
        List<Double> lambdas = Grid.logUniform(0.00000001, 0.1, 100).stream().sorted(comparator.reversed()).collect(Collectors.toList());

        for (double lambda: lambdas){
            ElasticNetLogisticTrainer trainer = ElasticNetLogisticTrainer.newBuilder(logisticRegression, dataSet)
                    .setEpsilon(0.01).setL1Ratio(1).setRegularization(lambda).build();

            System.out.println("=================================");
            System.out.println("lambda = "+lambda);
            StopWatch stopWatch = new StopWatch();
            stopWatch.start();
            trainer.optimize();
            System.out.println(stopWatch);
            System.out.println("training accuracy = "+ Accuracy.accuracy(logisticRegression,dataSet));
            System.out.println("test accuracy = "+ Accuracy.accuracy(logisticRegression,testSet));
            System.out.println("number of non-zeros= "+logisticRegression.getWeights().getAllWeights().getNumNonZeroElements());
        }

    }

    private static void test6() throws Exception{


        ClfDataSet dataSet = TRECFormat.loadClfDataSet(new File(DATASETS, "/amazon_book_genre/3/train.trec"),
                DataSetType.CLF_SPARSE, true);
        ClfDataSet testSet = TRECFormat.loadClfDataSet(new File(DATASETS, "/amazon_book_genre/3/test.trec"),
                DataSetType.CLF_SPARSE, true);
        System.out.println(dataSet.getMetaInfo());
        LogisticRegression logisticRegression = new LogisticRegression(dataSet.getNumClasses(),dataSet.getNumFeatures());
        ElasticNetLogisticTrainer trainer = ElasticNetLogisticTrainer.newBuilder(logisticRegression, dataSet)
                .setEpsilon(0.01).setL1Ratio(0).setRegularization(0.10000000000000006).build();

        StopWatch stopWatch = new StopWatch();
        stopWatch.start();
        trainer.optimize();
        System.out.println(stopWatch);
        System.out.println("training accuracy = "+ Accuracy.accuracy(logisticRegression,dataSet));
        System.out.println("test accuracy = "+ Accuracy.accuracy(logisticRegression,testSet));
        System.out.println("number of non-zeros= "+logisticRegression.getWeights().getAllWeights().getNumNonZeroElements());


    }

    private static void test7() throws Exception{
        ClfDataSet dataSet = TRECFormat.loadClfDataSet(new File(DATASETS, "20newsgroup/1/train.trec"),
                DataSetType.CLF_SPARSE, true);
        ClfDataSet testSet = TRECFormat.loadClfDataSet(new File(DATASETS, "20newsgroup/1/test.trec"),
                DataSetType.CLF_SPARSE, true);
        System.out.println(dataSet.getMetaInfo());
        LogisticRegression logisticRegression = new LogisticRegression(dataSet.getNumClasses(),dataSet.getNumFeatures());
        ElasticNetLogisticTrainer trainer = ElasticNetLogisticTrainer.newBuilder(logisticRegression, dataSet)
                .setEpsilon(0.01).setL1Ratio(0.1111111111111111).setRegularization(1.1233240329780266E-6).build();

        trainer.optimize();
        System.out.println("training accuracy = "+ Accuracy.accuracy(logisticRegression,dataSet));
        System.out.println("test accuracy = "+ Accuracy.accuracy(logisticRegression,testSet));


    }

}