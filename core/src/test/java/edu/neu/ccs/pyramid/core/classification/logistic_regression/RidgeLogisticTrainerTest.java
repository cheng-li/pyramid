package edu.neu.ccs.pyramid.core.classification.logistic_regression;

import edu.neu.ccs.pyramid.core.configuration.Config;
import edu.neu.ccs.pyramid.core.dataset.ClfDataSet;
import edu.neu.ccs.pyramid.core.dataset.DataSetType;
import edu.neu.ccs.pyramid.core.dataset.TRECFormat;
import edu.neu.ccs.pyramid.core.eval.Accuracy;
import org.apache.commons.lang3.time.StopWatch;
import org.apache.logging.log4j.Level;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.core.LoggerContext;
import org.apache.logging.log4j.core.config.Configuration;
import org.apache.logging.log4j.core.config.LoggerConfig;

import java.io.File;

public class RidgeLogisticTrainerTest {
    private static final Config config = new Config("config/local.properties");
    private static final String DATASETS = config.getString("input.datasets");
    private static final String TMP = config.getString("output.tmp");

    public static void main(String[] args) throws Exception{
        LoggerContext ctx = (LoggerContext) LogManager.getContext(false);
        Configuration config = ctx.getConfiguration();
        LoggerConfig loggerConfig = config.getLoggerConfig(LogManager.ROOT_LOGGER_NAME);
        loggerConfig.setLevel(Level.DEBUG);
        ctx.updateLoggers();
        test1();
    }

    private static void test1() throws Exception{
        ClfDataSet dataSet = TRECFormat.loadClfDataSet(new File(DATASETS, "/imdb/3/train.trec"),
                DataSetType.CLF_SPARSE, true);
        ClfDataSet testSet = TRECFormat.loadClfDataSet(new File(DATASETS, "/imdb/3/test.trec"),
                DataSetType.CLF_SPARSE, true);
        System.out.println(dataSet.getMetaInfo());
        RidgeLogisticTrainer trainer = RidgeLogisticTrainer.getBuilder()
                .setEpsilon(1)
                .setGaussianPriorVariance(0.5)
                .setHistory(5)
                .build();


        LogisticRegression logisticRegression = trainer.train(dataSet);
        System.out.println("train: "+ Accuracy.accuracy(logisticRegression, dataSet));
        System.out.println("test: "+Accuracy.accuracy(logisticRegression,testSet));
    }

    private static void test2() throws Exception{
        ClfDataSet dataSet = TRECFormat.loadClfDataSet(new File(DATASETS, "/spam/trec_data/train.trec"),
                DataSetType.CLF_SPARSE, true);
        ClfDataSet testSet = TRECFormat.loadClfDataSet(new File(DATASETS, "/spam/trec_data/test.trec"),
                DataSetType.CLF_SPARSE, true);
        System.out.println(dataSet.getMetaInfo());
        RidgeLogisticTrainer trainer = RidgeLogisticTrainer.getBuilder()
                .setEpsilon(0.001)
                .setGaussianPriorVariance(10000)
                .setHistory(5)
                .build();


        LogisticRegression logisticRegression = trainer.train(dataSet);
        System.out.println("train: "+ Accuracy.accuracy(logisticRegression, dataSet));
        System.out.println("test: "+Accuracy.accuracy(logisticRegression,testSet));
    }


    private static void test3() throws Exception{
        ClfDataSet dataSet = TRECFormat.loadClfDataSet(new File(DATASETS, "20newsgroup/1/train.trec"),
                DataSetType.CLF_SPARSE, true);
        ClfDataSet testSet = TRECFormat.loadClfDataSet(new File(DATASETS, "20newsgroup/1/test.trec"),
                DataSetType.CLF_SPARSE, true);
        System.out.println(dataSet.getMetaInfo());
        RidgeLogisticTrainer trainer = RidgeLogisticTrainer.getBuilder()
                .setEpsilon(0.01)
                .setGaussianPriorVariance(0.5)
                .setHistory(5)
                .build();

        StopWatch stopWatch = new StopWatch();
        stopWatch.start();
        LogisticRegression logisticRegression = trainer.train(dataSet);
        System.out.println(stopWatch);
        System.out.println("train: "+ Accuracy.accuracy(logisticRegression, dataSet));
        System.out.println("test: "+Accuracy.accuracy(logisticRegression,testSet));
    }

}