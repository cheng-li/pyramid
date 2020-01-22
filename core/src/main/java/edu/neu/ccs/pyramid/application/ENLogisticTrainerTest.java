package edu.neu.ccs.pyramid.application;

import com.google.common.base.Stopwatch;
import edu.neu.ccs.pyramid.classification.logistic_regression.ElasticNetBinaryLogisticTrainer;
import edu.neu.ccs.pyramid.classification.logistic_regression.ElasticNetLogisticTrainer;
import edu.neu.ccs.pyramid.classification.logistic_regression.LogisticRegression;
import edu.neu.ccs.pyramid.configuration.Config;
import edu.neu.ccs.pyramid.dataset.ClfDataSet;
import edu.neu.ccs.pyramid.dataset.DataSetType;
import edu.neu.ccs.pyramid.dataset.TRECFormat;
import edu.neu.ccs.pyramid.eval.Accuracy;

import java.io.File;

/**
 *
 * testing the ElasticNet Trainer for either general logistic regression or binary logistic regression
 * Created by Rainicy on 1/21/20
 */
public class ENLogisticTrainerTest {
    public static void main(String[] args) throws Exception {
        if (args.length != 1){
            throw new IllegalArgumentException("Please specify a properties file.");
        }

        Config config = new Config(args[0]);

        System.out.println(config);

        String lrType = config.getString("logistic_regression_type");

        switch (lrType) {
            case "general": {
                System.out.println("testing general");
                general_test(config);
                break;
            }
            case "binary": {
                System.out.println("testing binary");
                binary_test(config);
                break;
            }
            default: break;
        }
    }


    /**
     * Test for ElasticNet optimizer for  general logistic regression.
     * @throws Exception
     */
    private static void general_test(Config config) throws Exception{
        // TODO: test clf_sparse and clf_seqsparse
        ClfDataSet dataSet = TRECFormat.loadClfDataSet(new File(config.getString("input.train")),
                DataSetType.CLF_SPARSE, true);
        ClfDataSet testSet = TRECFormat.loadClfDataSet(new File(config.getString("input.test")),
                DataSetType.CLF_SPARSE, true);
        System.out.println(dataSet.getMetaInfo());
        LogisticRegression logisticRegression = new LogisticRegression(dataSet.getNumClasses(),dataSet.getNumFeatures());
        ElasticNetLogisticTrainer trainer = ElasticNetLogisticTrainer.newBuilder(logisticRegression, dataSet)
                .setL1Ratio(config.getDouble("l1_ratio"))
                .setRegularization(config.getDouble("regularization"))
                .setLineSearch(config.getBoolean("line_search"))
                .setMaxNumLinearRegUpdates(config.getInt("max_iter_linear_regression")).build();
        Stopwatch stopwatch = Stopwatch.createStarted();
        trainer.optimize();
        long used  = Runtime.getRuntime().totalMemory() - Runtime.getRuntime().freeMemory();
        stopwatch.stop();
        System.out.println("used memory= " + used/1024/1024 + " Mb");
        System.out.println("total time = " + stopwatch);
        System.out.println("training accuracy = "+ Accuracy.accuracy(logisticRegression,dataSet));
        System.out.println("test accuracy = "+ Accuracy.accuracy(logisticRegression,testSet));
    }

    /**
     * Test for ElasticNet optimizer for  general logistic regression.
     * @throws Exception
     */
    private static void binary_test(Config config) throws Exception{
        ClfDataSet dataSet = TRECFormat.loadClfDataSet(new File(config.getString("input.train")),
                DataSetType.CLF_SPARSE, true);
        ClfDataSet testSet = TRECFormat.loadClfDataSet(new File(config.getString("input.test")),
                DataSetType.CLF_SPARSE, true);
        System.out.println(dataSet.getMetaInfo());
        LogisticRegression lR = new LogisticRegression(1, dataSet.getNumFeatures(), false, true);
        ElasticNetBinaryLogisticTrainer trainer = ElasticNetBinaryLogisticTrainer.newBuilder(lR, dataSet)
                .setL1Ratio(config.getDouble("l1_ratio"))
                .setRegularization(config.getDouble("regularization"))
                .setLineSearch(config.getBoolean("line_search"))
                .setMaxNumLinearRegUpdates(config.getInt("max_iter_linear_regression")).build();
        Stopwatch stopwatch = Stopwatch.createStarted();
        trainer.optimize();
        long used  = Runtime.getRuntime().totalMemory() - Runtime.getRuntime().freeMemory();
        stopwatch.stop();
        System.out.println("used memory= " + used/1024/1024 + " Mb");
        System.out.println("total time = " + stopwatch);
        System.out.println("training accuracy = "+ Accuracy.accuracy(lR,dataSet));
        System.out.println("test accuracy = "+ Accuracy.accuracy(lR,testSet));
    }
}
