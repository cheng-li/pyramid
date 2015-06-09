package edu.neu.ccs.pyramid.classification.logistic_regression;

import edu.neu.ccs.pyramid.configuration.Config;
import edu.neu.ccs.pyramid.dataset.ClfDataSet;
import edu.neu.ccs.pyramid.dataset.DataSetType;
import edu.neu.ccs.pyramid.dataset.TRECFormat;
import edu.neu.ccs.pyramid.eval.Accuracy;
import edu.neu.ccs.pyramid.optimization.ConjugateGradientDescent;
import edu.neu.ccs.pyramid.optimization.GradientDescent;
import edu.neu.ccs.pyramid.optimization.LBFGS;

import java.io.File;

public class LogisticRegressionTest {
    private static final Config config = new Config("configs/local.config");
    private static final String DATASETS = config.getString("input.datasets");
    private static final String TMP = config.getString("output.tmp");

    public static void main(String[] args) throws Exception{
        test1();

    }

    private static void test1() throws Exception{

        ClfDataSet dataSet = TRECFormat.loadClfDataSet(new File(DATASETS, "/imdb/3/train.trec"),
                DataSetType.CLF_SPARSE, false);
        ClfDataSet testSet = TRECFormat.loadClfDataSet(new File(DATASETS, "/imdb/3/test.trec"),
                DataSetType.CLF_SPARSE, false);
        System.out.println(dataSet.getMetaInfo());

        LogisticRegression logisticRegression = new LogisticRegression(dataSet.getNumClasses(),dataSet.getNumFeatures());
        logisticRegression.setFeatureExtraction(true);
        LogisticLoss function = new LogisticLoss(logisticRegression,dataSet,1000);
        GradientDescent gradientDescent = new GradientDescent(function,1);
        for (int i=0;i<500;i++){
            gradientDescent.iterate();
            System.out.println(Accuracy.accuracy(logisticRegression,dataSet));
        }

    }


    private static void test2() throws Exception{

        ClfDataSet dataSet = TRECFormat.loadClfDataSet(new File(DATASETS, "/imdb/train.trec"),
                DataSetType.CLF_SPARSE, true);
        ClfDataSet testSet = TRECFormat.loadClfDataSet(new File(DATASETS, "/imdb/test.trec"),
                DataSetType.CLF_SPARSE, true);
        System.out.println(dataSet.getMetaInfo());

        LogisticRegression logisticRegression = new LogisticRegression(dataSet.getNumClasses(),dataSet.getNumFeatures());
        logisticRegression.setFeatureExtraction(true);
        LogisticLoss function = new LogisticLoss(logisticRegression,dataSet,0.1);
        ConjugateGradientDescent conjugateGradientDescent = new ConjugateGradientDescent(function,0.001);
        for (int i=0;i<100;i++){
            System.out.println("--------");
            System.out.println("iteration "+i);
            conjugateGradientDescent.update();
            System.out.println("loss: " + function.getValue());
            System.out.println("train: "+Accuracy.accuracy(logisticRegression,dataSet));
            System.out.println("test: "+Accuracy.accuracy(logisticRegression,testSet));
        }

    }


    private static void test3() throws Exception{

        ClfDataSet dataSet = TRECFormat.loadClfDataSet(new File(DATASETS, "/imdb/train.trec"),
                DataSetType.CLF_SPARSE, true);
        ClfDataSet testSet = TRECFormat.loadClfDataSet(new File(DATASETS, "/imdb/test.trec"),
                DataSetType.CLF_SPARSE, true);
        System.out.println(dataSet.getMetaInfo());

        LogisticRegression logisticRegression = new LogisticRegression(dataSet.getNumClasses(),dataSet.getNumFeatures());
        logisticRegression.setFeatureExtraction(false);
        LogisticLoss function = new LogisticLoss(logisticRegression,dataSet,0.1);
        LBFGS lbfgs = new LBFGS(function);
        for (int i=0;i<20;i++){
            System.out.println("--------");
            System.out.println("iteration "+i);
            lbfgs.iterate();
            System.out.println("loss: " + function.getValue());
            System.out.println("train: "+Accuracy.accuracy(logisticRegression,dataSet));
            System.out.println("test: "+Accuracy.accuracy(logisticRegression,testSet));
        }

    }

    private static void test4() throws Exception{

        ClfDataSet dataSet = TRECFormat.loadClfDataSet(new File(DATASETS, "/imdb/3/train.trec"),
                DataSetType.CLF_SPARSE, false);
        ClfDataSet testSet = TRECFormat.loadClfDataSet(new File(DATASETS, "/imdb/3/test.trec"),
                DataSetType.CLF_SPARSE, false);
        System.out.println(dataSet.getMetaInfo());

        LogisticRegression logisticRegression = new LogisticRegression(dataSet.getNumClasses(),dataSet.getNumFeatures());
        logisticRegression.setFeatureExtraction(false);
        LogisticLoss function = new LogisticLoss(logisticRegression,dataSet,0.1);
        LBFGS lbfgs = new LBFGS(function);
        lbfgs.optimize();
        System.out.println("train: "+Accuracy.accuracy(logisticRegression,dataSet));
        System.out.println("test: "+Accuracy.accuracy(logisticRegression,testSet));

    }

}