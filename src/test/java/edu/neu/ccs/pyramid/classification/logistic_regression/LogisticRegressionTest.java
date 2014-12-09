package edu.neu.ccs.pyramid.classification.logistic_regression;

import edu.neu.ccs.pyramid.configuration.Config;
import edu.neu.ccs.pyramid.dataset.ClfDataSet;
import edu.neu.ccs.pyramid.dataset.DataSetType;
import edu.neu.ccs.pyramid.dataset.TRECFormat;
import edu.neu.ccs.pyramid.eval.Accuracy;
import edu.neu.ccs.pyramid.optimization.ConjugateGradientDescent;
import edu.neu.ccs.pyramid.optimization.GradientDescent;

import java.io.File;

import static org.junit.Assert.*;

public class LogisticRegressionTest {
    private static final Config config = new Config("configs/local.config");
    private static final String DATASETS = config.getString("input.datasets");
    private static final String TMP = config.getString("output.tmp");

    public static void main(String[] args) throws Exception{
        test2();

    }

    private static void test1() throws Exception{

        ClfDataSet dataSet = TRECFormat.loadClfDataSet(new File(DATASETS, "/imdb/train.trec"),
                DataSetType.CLF_SPARSE, true);
        System.out.println(dataSet.getMetaInfo());

        LogisticRegression logisticRegression = new LogisticRegression(dataSet.getNumClasses(),dataSet.getNumFeatures());
        logisticRegression.setFeatureExtraction(true);
        LogisticLoss function = new LogisticLoss(logisticRegression,dataSet,1000);
        GradientDescent gradientDescent = new GradientDescent(function,1);
        for (int i=0;i<500;i++){
            gradientDescent.update();
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
            System.out.println("loss: " + function.getValue(logisticRegression.getWeights().getAllWeights()));
            System.out.println("train: "+Accuracy.accuracy(logisticRegression,dataSet));
            System.out.println("test: "+Accuracy.accuracy(logisticRegression,testSet));
        }

    }

}