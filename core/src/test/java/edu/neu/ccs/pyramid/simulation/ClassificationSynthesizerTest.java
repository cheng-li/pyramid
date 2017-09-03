package edu.neu.ccs.pyramid.simulation;

import edu.neu.ccs.pyramid.classification.logistic_regression.LogisticRegression;
import edu.neu.ccs.pyramid.classification.logistic_regression.RidgeLogisticTrainer;
import edu.neu.ccs.pyramid.configuration.Config;
import edu.neu.ccs.pyramid.dataset.ClfDataSet;
import edu.neu.ccs.pyramid.dataset.TRECFormat;
import edu.neu.ccs.pyramid.eval.Accuracy;

import java.io.File;

import static org.junit.Assert.*;

public class ClassificationSynthesizerTest {
    private static final Config config = new Config("config/local.properties");
    private static final String DATASETS = config.getString("input.datasets");
    private static final String TMP = config.getString("output.tmp");

    public static void main(String[] args) {
        test1();
        test2();
        test3();
        test4();
        test5();
        test6();
    }

    private static void test1(){

        ClassificationSynthesizer classificationSynthesizer = ClassificationSynthesizer.getBuilder()
                .setNumDataPoints(1000)
                .setNumFeatures(2)
                .setNoiseSD(0.00000001)
                .build();
        ClfDataSet trainSet = classificationSynthesizer.multivarLine();
        ClfDataSet testSet = classificationSynthesizer.multivarLine();
        TRECFormat.save(trainSet, new File(TMP, "line1/train.trec"));
        TRECFormat.save(testSet,new File(TMP,"line1/test.trec"));

        RidgeLogisticTrainer trainer = RidgeLogisticTrainer.getBuilder().setGaussianPriorVariance(1).build();
        LogisticRegression logisticRegression = trainer.train(trainSet);
        System.out.println(Accuracy.accuracy(logisticRegression,trainSet));
        System.out.println(Accuracy.accuracy(logisticRegression,testSet));
        System.out.println(logisticRegression.getWeights().getWeightsForClass(0));
    }

    private static void test2(){

        ClassificationSynthesizer classificationSynthesizer = ClassificationSynthesizer.getBuilder()
                .setNumDataPoints(100)
                .setNumFeatures(2)
                .setNoiseSD(0.00000001)
                .build();
        ClfDataSet trainSet = classificationSynthesizer.multivarLine();
        ClfDataSet testSet = classificationSynthesizer.multivarLine();
        TRECFormat.save(trainSet, new File(TMP, "line2/train.trec"));
        TRECFormat.save(testSet,new File(TMP,"line2/test.trec"));

        RidgeLogisticTrainer trainer = RidgeLogisticTrainer.getBuilder().setGaussianPriorVariance(1).build();
        LogisticRegression logisticRegression = trainer.train(trainSet);
        System.out.println(Accuracy.accuracy(logisticRegression,trainSet));
        System.out.println(Accuracy.accuracy(logisticRegression,testSet));
        System.out.println(logisticRegression.getWeights().getWeightsForClass(0));
    }


    private static void test3(){

        ClassificationSynthesizer classificationSynthesizer = ClassificationSynthesizer.getBuilder()
                .setNumDataPoints(1000)
                .setNumFeatures(2)
                .setNoiseSD(0.1)
                .build();
        ClfDataSet trainSet = classificationSynthesizer.multivarLine();
        ClfDataSet testSet = classificationSynthesizer.multivarLine();
        TRECFormat.save(trainSet, new File(TMP, "line3/train.trec"));
        TRECFormat.save(testSet,new File(TMP,"line3/test.trec"));

        RidgeLogisticTrainer trainer = RidgeLogisticTrainer.getBuilder().setGaussianPriorVariance(1).build();
        LogisticRegression logisticRegression = trainer.train(trainSet);
        System.out.println(Accuracy.accuracy(logisticRegression,trainSet));
        System.out.println(Accuracy.accuracy(logisticRegression,testSet));
        System.out.println(logisticRegression.getWeights().getWeightsForClass(0));
    }

    private static void test4(){

        ClassificationSynthesizer classificationSynthesizer = ClassificationSynthesizer.getBuilder()
                .setNumDataPoints(1000)
                .setNumFeatures(3)
                .setNoiseSD(0.00000001)
                .build();
        ClfDataSet trainSet = classificationSynthesizer.multivarLine();
        ClfDataSet testSet = classificationSynthesizer.multivarLine();
        TRECFormat.save(trainSet, new File(TMP, "line4/train.trec"));
        TRECFormat.save(testSet,new File(TMP,"line4/test.trec"));

        RidgeLogisticTrainer trainer = RidgeLogisticTrainer.getBuilder().setGaussianPriorVariance(1).build();
        LogisticRegression logisticRegression = trainer.train(trainSet);
        System.out.println(Accuracy.accuracy(logisticRegression,trainSet));
        System.out.println(Accuracy.accuracy(logisticRegression,testSet));
        System.out.println(logisticRegression.getWeights().getWeightsForClass(0));
    }

    private static void test5(){

        ClassificationSynthesizer classificationSynthesizer = ClassificationSynthesizer.getBuilder()
                .setNumDataPoints(1000)
                .setNumFeatures(10)
                .setNoiseSD(0.00000001)
                .build();
        ClfDataSet trainSet = classificationSynthesizer.multivarLine();
        ClfDataSet testSet = classificationSynthesizer.multivarLine();
        TRECFormat.save(trainSet, new File(TMP, "line5/train.trec"));
        TRECFormat.save(testSet,new File(TMP,"line5/test.trec"));

        RidgeLogisticTrainer trainer = RidgeLogisticTrainer.getBuilder().setGaussianPriorVariance(1).build();
        LogisticRegression logisticRegression = trainer.train(trainSet);
        System.out.println(Accuracy.accuracy(logisticRegression,trainSet));
        System.out.println(Accuracy.accuracy(logisticRegression,testSet));
        System.out.println(logisticRegression.getWeights().getWeightsForClass(0));
    }

    private static void test6(){

        ClassificationSynthesizer classificationSynthesizer = ClassificationSynthesizer.getBuilder()
                .setNumDataPoints(1000)
                .setNumFeatures(20)
                .setNoiseSD(0.00000001)
                .build();
        ClfDataSet trainSet = classificationSynthesizer.multivarLine();
        ClfDataSet testSet = classificationSynthesizer.multivarLine();
        TRECFormat.save(trainSet, new File(TMP, "line6/train.trec"));
        TRECFormat.save(testSet,new File(TMP,"line6/test.trec"));

        RidgeLogisticTrainer trainer = RidgeLogisticTrainer.getBuilder().setGaussianPriorVariance(1).build();
        LogisticRegression logisticRegression = trainer.train(trainSet);
        System.out.println(Accuracy.accuracy(logisticRegression,trainSet));
        System.out.println(Accuracy.accuracy(logisticRegression,testSet));
        System.out.println(logisticRegression.getWeights().getWeightsForClass(0));
    }

}