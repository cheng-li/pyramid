package edu.neu.ccs.pyramid.experiment;

import edu.neu.ccs.pyramid.classification.logistic_regression.ElasticNetLogisticTrainer;
import edu.neu.ccs.pyramid.classification.logistic_regression.LogisticRegression;
import edu.neu.ccs.pyramid.classification.logistic_regression.LogisticRegressionInspector;
import edu.neu.ccs.pyramid.classification.logistic_regression.RidgeLogisticTrainer;
import edu.neu.ccs.pyramid.configuration.Config;
import edu.neu.ccs.pyramid.dataset.ClfDataSet;
import edu.neu.ccs.pyramid.dataset.DataSetType;
import edu.neu.ccs.pyramid.dataset.TRECFormat;
import edu.neu.ccs.pyramid.eval.*;
import edu.neu.ccs.pyramid.util.Grid;
import org.apache.commons.lang3.time.StopWatch;

import java.io.File;
import java.util.Comparator;
import java.util.List;
import java.util.stream.Collectors;

/**
 * elastic-net logistic regression
 * Created by chengli on 2/25/15.
 */
public class Exp70 {
    public static void main(String[] args) throws Exception{
        if (args.length !=1){
            throw new IllegalArgumentException("please specify the config file");
        }

        Config config = new Config(args[0]);
        System.out.println(config);


        train(config);

    }

    private static void train(Config config) throws Exception{
        String input = config.getString("input.folder");
        ClfDataSet dataSet = TRECFormat.loadClfDataSet(new File(input, "train.trec"),
                DataSetType.CLF_SPARSE, true);
        ClfDataSet testSet = TRECFormat.loadClfDataSet(new File(input,"test.trec"),
                DataSetType.CLF_SPARSE, true);
        System.out.println(dataSet.getMetaInfo());



        Comparator<Double> comparator = Comparator.comparing(Double::doubleValue);

        List<Double> l1Ratios = Grid.uniform(config.getDouble("train.l1Ratio.min"),
                config.getDouble("train.l1Ratio.max"), config.getInt("train.l1Ratio.size"))
                .stream().sorted().collect(Collectors.toList());

        List<Double> lambdas = Grid.logUniform(config.getDouble("train.lambda.min"),
                config.getDouble("train.lambda.max") , config.getInt("train.lambda.size"))
                .stream().sorted(comparator.reversed()).collect(Collectors.toList());

        double bestAcc=0;

        for (double l1ratio: l1Ratios){
            LogisticRegression logisticRegression = new LogisticRegression(dataSet.getNumClasses(),dataSet.getNumFeatures());
            for (double lambda: lambdas){
                ElasticNetLogisticTrainer trainer = ElasticNetLogisticTrainer.getBuilder()
                        .setEpsilon(0.01).setL1Ratio(l1ratio).setRegularization(lambda).build();

                System.out.println("=================================");
                System.out.println("l1Ratio= "+l1ratio );
                System.out.println("lambda = "+lambda);
                StopWatch stopWatch = new StopWatch();
                stopWatch.start();
                trainer.train(logisticRegression, dataSet);
                System.out.println("time spent = " + stopWatch);
                System.out.println("training accuracy = "+ Accuracy.accuracy(logisticRegression,dataSet));
                double acc = Accuracy.accuracy(logisticRegression, testSet);
                System.out.println("test accuracy = "+ acc);
                System.out.println("number of used features = "+ LogisticRegressionInspector.numOfUsedFeatures(logisticRegression));

                if (acc> bestAcc){
                    bestAcc = acc;
                    System.out.println("*********************************************");
                    System.out.println("best test accuracy got so far = "+bestAcc);
                    System.out.println("l1Ratio= "+l1ratio );
                    System.out.println("lambda = "+lambda);
                    System.out.println("number of used features = "+ LogisticRegressionInspector.numOfUsedFeatures(logisticRegression));
                    System.out.println("*********************************************");
                }

            }
        }


    }
}
