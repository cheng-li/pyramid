package edu.neu.ccs.pyramid.classification.logistic_regression;

import edu.neu.ccs.pyramid.configuration.Config;
import edu.neu.ccs.pyramid.dataset.ClfDataSet;
import edu.neu.ccs.pyramid.dataset.DataSetType;
import edu.neu.ccs.pyramid.dataset.TRECFormat;
import edu.neu.ccs.pyramid.eval.Accuracy;

import java.io.File;

import static org.junit.Assert.*;

public class ElasticNetLogisticTrainerTest {
    private static final Config config = new Config("configs/local.config");
    private static final String DATASETS = config.getString("input.datasets");
    private static final String TMP = config.getString("output.tmp");
    public static void main(String[] args) throws Exception{
        test3();
    }

    private static void test1() throws Exception{
        ClfDataSet dataSet = TRECFormat.loadClfDataSet(new File(DATASETS, "/spam/trec_data/train.trec"),
                DataSetType.CLF_SPARSE, true);
        ClfDataSet testSet = TRECFormat.loadClfDataSet(new File(DATASETS, "/spam/trec_data/test.trec"),
                DataSetType.CLF_SPARSE, true);
        System.out.println(dataSet.getMetaInfo());
        LogisticRegression logisticRegression = new LogisticRegression(dataSet.getNumClasses(),dataSet.getNumFeatures());
        ElasticNetLogisticTrainer trainer = ElasticNetLogisticTrainer.getBuilder()
                .setEpsilon(0.01).setL1Ratio(0.5).setRegularization(0.01).build();
        for (int i=0;i<10;i++){
            System.out.println("iteration "+i);
            trainer.iterate(logisticRegression,dataSet);
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
        ElasticNetLogisticTrainer trainer = ElasticNetLogisticTrainer.getBuilder()
                .setEpsilon(0.01).setL1Ratio(0.5).setRegularization(0.01).build();

        trainer.train(logisticRegression,dataSet);
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
        ElasticNetLogisticTrainer trainer = ElasticNetLogisticTrainer.getBuilder()
                .setEpsilon(0.01).setL1Ratio(0.5).setRegularization(0.01).build();

        for (int i=0;i<10;i++){
            System.out.println("iteration "+i);
            trainer.iterate(logisticRegression,dataSet);
            System.out.println("training accuracy = "+ Accuracy.accuracy(logisticRegression,dataSet));
            System.out.println("test accuracy = "+ Accuracy.accuracy(logisticRegression,testSet));
        }


    }

}