package edu.neu.ccs.pyramid.classification.logistic_regression;

import edu.neu.ccs.pyramid.configuration.Config;
import edu.neu.ccs.pyramid.dataset.ClfDataSet;
import edu.neu.ccs.pyramid.dataset.DataSetType;
import edu.neu.ccs.pyramid.dataset.TRECFormat;
import edu.neu.ccs.pyramid.eval.Accuracy;

import java.io.File;

import static org.junit.Assert.*;

public class RidgeLogisticOptimizerTest {
    private static final Config config = new Config("config/local.config");
    private static final String DATASETS = config.getString("input.datasets");
    private static final String TMP = config.getString("output.tmp");
    public static void main(String[] args) throws Exception{
        test1();
    }

    private static void test1() throws Exception{
        ClfDataSet dataSet = TRECFormat.loadClfDataSet(new File(DATASETS, "/spam/trec_data/train.trec"),
                DataSetType.CLF_SPARSE, true);
        ClfDataSet testSet = TRECFormat.loadClfDataSet(new File(DATASETS, "/spam/trec_data/test.trec"),
                DataSetType.CLF_SPARSE, true);
        LogisticRegression logisticRegression = new LogisticRegression(dataSet.getNumClasses(),dataSet.getNumFeatures());
        RidgeLogisticOptimizer optimizer = new RidgeLogisticOptimizer(0.5);
        System.out.println("after initialization");
        System.out.println("train acc = " + Accuracy.accuracy(logisticRegression, dataSet));
        System.out.println("test acc = "+Accuracy.accuracy(logisticRegression,testSet));
        optimizer.optimize(logisticRegression,dataSet);
        System.out.println("after training");
        System.out.println("train acc = " + Accuracy.accuracy(logisticRegression, dataSet));
        System.out.println("test acc = "+Accuracy.accuracy(logisticRegression,testSet));
    }
}