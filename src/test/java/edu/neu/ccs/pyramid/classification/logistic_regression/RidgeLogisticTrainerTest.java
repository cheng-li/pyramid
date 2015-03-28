package edu.neu.ccs.pyramid.classification.logistic_regression;

import edu.neu.ccs.pyramid.configuration.Config;
import edu.neu.ccs.pyramid.dataset.ClfDataSet;
import edu.neu.ccs.pyramid.dataset.DataSetType;
import edu.neu.ccs.pyramid.dataset.TRECFormat;
import edu.neu.ccs.pyramid.eval.Accuracy;

import java.io.File;

import static org.junit.Assert.*;

public class RidgeLogisticTrainerTest {
    private static final Config config = new Config("configs/local.config");
    private static final String DATASETS = config.getString("input.datasets");
    private static final String TMP = config.getString("output.tmp");

    public static void main(String[] args) throws Exception{
        test3();
    }

    private static void test1() throws Exception{
        ClfDataSet dataSet = TRECFormat.loadClfDataSet(new File(DATASETS, "/imdb/train.trec"),
                DataSetType.CLF_SPARSE, true);
        ClfDataSet testSet = TRECFormat.loadClfDataSet(new File(DATASETS, "/imdb/test.trec"),
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
                .setEpsilon(0.01)
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
                .setEpsilon(1)
                .setGaussianPriorVariance(0.5)
                .setHistory(5)
                .build();

        LogisticRegression logisticRegression = trainer.train(dataSet);
        System.out.println("train: "+ Accuracy.accuracy(logisticRegression, dataSet));
        System.out.println("test: "+Accuracy.accuracy(logisticRegression,testSet));
    }

}