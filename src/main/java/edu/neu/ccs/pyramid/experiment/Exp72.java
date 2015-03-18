package edu.neu.ccs.pyramid.experiment;

import edu.neu.ccs.pyramid.configuration.Config;
import edu.neu.ccs.pyramid.dataset.*;
import edu.neu.ccs.pyramid.eval.Accuracy;
import edu.neu.ccs.pyramid.eval.Overlap;

import edu.neu.ccs.pyramid.multilabel_classification.adaboostmh.AdaBoostMH;
import edu.neu.ccs.pyramid.multilabel_classification.adaboostmh.AdaBoostMHTrainer;

import org.apache.commons.lang3.time.StopWatch;

import java.io.File;


/**
 * Created by chengli on 3/18/15.
 */
public class Exp72 {
    public static void main(String[] args) throws Exception{
        if (args.length !=1){
            throw new IllegalArgumentException("please specify the config file");
        }

        Config config = new Config(args[0]);
        System.out.println(config);

        File output = new File(config.getString("output.folder"));
        output.mkdirs();


        if (config.getBoolean("train")){
            train(config);
        }



    }

    static MultiLabelClfDataSet loadTrainData(Config config) throws Exception{
        String trainFile = new File(config.getString("input.folder"),
                config.getString("input.trainData")).getAbsolutePath();
        MultiLabelClfDataSet dataSet;

        if (config.getBoolean("input.featureMatrix.sparse")){
            dataSet= TRECFormat.loadMultiLabelClfDataSet(new File(trainFile), DataSetType.ML_CLF_SPARSE,
                    true);
        } else {
            dataSet= TRECFormat.loadMultiLabelClfDataSet(new File(trainFile), DataSetType.ML_CLF_DENSE,
                    true);
        }

        return dataSet;
    }

    static MultiLabelClfDataSet loadTestData(Config config) throws Exception{
        String trainFile = new File(config.getString("input.folder"),
                config.getString("input.testData")).getAbsolutePath();
        MultiLabelClfDataSet dataSet;

        if (config.getBoolean("input.featureMatrix.sparse")){
            dataSet= TRECFormat.loadMultiLabelClfDataSet(new File(trainFile), DataSetType.ML_CLF_SPARSE,
                    true);
        } else {
            dataSet= TRECFormat.loadMultiLabelClfDataSet(new File(trainFile), DataSetType.ML_CLF_DENSE,
                    true);
        }

        return dataSet;
    }

    static void train(Config config) throws Exception{
        String output = config.getString("output.folder");
        int numIterations = config.getInt("train.numIterations");
        String modelName = config.getString("output.model");
        StopWatch stopWatch = new StopWatch();
        stopWatch.start();

        MultiLabelClfDataSet dataSet = loadTrainData(config);
        MultiLabelClfDataSet testDataSet = loadTestData(config);


        int numClasses = dataSet.getNumClasses();
        System.out.println("number of class = "+numClasses);

        AdaBoostMH boosting;
        if (config.getBoolean("train.warmStart")){
            boosting = AdaBoostMH.deserialize(new File(output,modelName));
        } else {
            boosting = new AdaBoostMH(numClasses);
        }

        AdaBoostMHTrainer trainer = new AdaBoostMHTrainer(dataSet,boosting);

        for (int i=0;i<numIterations;i++){
            System.out.println("iteration "+i);
            trainer.iterate();
            if (config.getBoolean("train.showPerformanceEachRound")){
                System.out.println("model size = "+boosting.getRegressors(0).size());
                System.out.println("accuracy on training set = "+ Accuracy.accuracy(boosting,
                        dataSet));
                System.out.println("overlap on training set = "+ Overlap.overlap(boosting, dataSet));

                System.out.println("accuracy on test set = "+ Accuracy.accuracy(boosting,
                        testDataSet));
                System.out.println("overlap on test set = "+ Overlap.overlap(boosting,testDataSet));
            }

        }
        File serializedModel =  new File(output,modelName);
        boosting.serialize(serializedModel);
        System.out.println(stopWatch);
    }

    static void verify(Config config) throws Exception{
        String output = config.getString("output.folder");
        String modelName = config.getString("output.model");
        AdaBoostMH boosting = AdaBoostMH.deserialize(new File(output,modelName));
        MultiLabelClfDataSet dataSet = loadTrainData(config);
        System.out.println("accuracy on training set = "+Accuracy.accuracy(boosting,dataSet));
    }

    static void test(Config config) throws Exception{
        String output = config.getString("output.folder");
        String modelName = config.getString("output.model");

        AdaBoostMH boosting = AdaBoostMH.deserialize(new File(output,modelName));
        MultiLabelClfDataSet dataSet = loadTestData(config);
        System.out.println("accuracy on test set = "+Accuracy.accuracy(boosting,dataSet));
        System.out.println("overlap on test set = "+ Overlap.overlap(boosting,dataSet));
    }
}
