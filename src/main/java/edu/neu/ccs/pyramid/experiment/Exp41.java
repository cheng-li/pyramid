package edu.neu.ccs.pyramid.experiment;

import edu.neu.ccs.pyramid.configuration.Config;
import edu.neu.ccs.pyramid.dataset.*;
import edu.neu.ccs.pyramid.eval.Accuracy;
import edu.neu.ccs.pyramid.eval.Overlap;
import edu.neu.ccs.pyramid.eval.PerClassMeasures;
import edu.neu.ccs.pyramid.multilabel_classification.multi_label_logistic_regression.MLLogisticRegression;
import edu.neu.ccs.pyramid.multilabel_classification.multi_label_logistic_regression.MLLogisticTrainer;
import org.apache.commons.lang3.time.StopWatch;
import org.apache.mahout.math.Vector;

import java.io.File;
import java.util.Arrays;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import java.util.stream.Collectors;

/**
 * multi-label logistic regression
 * Created by chengli on 12/23/14.
 */
public class Exp41 {
    public static void main(String[] args) throws Exception{
        if (args.length !=1){
            throw new IllegalArgumentException("please specify the config file");
        }

        Config config = new Config(args[0]);
        System.out.println(config);

        if (config.getBoolean("train")){
            train(config);
        }

        if (config.getBoolean("test")){
            test(config);
        }



    }

    static MultiLabelClfDataSet loadTrainData(Config config) throws Exception{
        String trainFile = new File(config.getString("input.folder"),
                config.getString("input.trainData")).getAbsolutePath();
        MultiLabelClfDataSet dataSet;
        dataSet= TRECFormat.loadMultiLabelClfDataSet(new File(trainFile), DataSetType.ML_CLF_SPARSE,
                    true);
        return dataSet;
    }

    static MultiLabelClfDataSet loadTestData(Config config) throws Exception{
        String trainFile = new File(config.getString("input.folder"),
                config.getString("input.testData")).getAbsolutePath();
        MultiLabelClfDataSet dataSet;
        dataSet= TRECFormat.loadMultiLabelClfDataSet(new File(trainFile), DataSetType.ML_CLF_SPARSE,
                    true);

        return dataSet;
    }

    static void train(Config config) throws Exception{
        String archive = config.getString("archive.folder");
        String modelName = config.getString("archive.model");

        StopWatch stopWatch = new StopWatch();
        stopWatch.start();

        MultiLabelClfDataSet dataSet = loadTrainData(config);
        System.out.println("training data set loaded");
        System.out.println(dataSet.getMetaInfo());
        List<MultiLabel> assignments = DataSetUtil.gatherLabels(dataSet).stream()
                .collect(Collectors.toList());
        MLLogisticTrainer trainer = MLLogisticTrainer.getBuilder().setEpsilon(config.getDouble("train.epsilon"))
                .setGaussianPriorVariance(config.getDouble("train.gaussianPriorVariance"))
                .setHistory(5).build();
        MLLogisticRegression mlLogisticRegression =trainer.train(dataSet,assignments);
        File serializedModel =  new File(archive,modelName);
        mlLogisticRegression.serialize(serializedModel);
        System.out.println("time spent = " + stopWatch);
        System.out.println("accuracy on training set = " +Accuracy.accuracy(mlLogisticRegression,dataSet));
        System.out.println("overlap on training set = "+ Overlap.overlap(mlLogisticRegression,dataSet));
    }

    static void test(Config config) throws Exception{
        String archive = config.getString("archive.folder");
        String modelName = config.getString("archive.model");

        MLLogisticRegression mlLogisticRegression = MLLogisticRegression.deserialize(new File(archive,modelName));
        System.out.println("test data set loaded");
        MultiLabelClfDataSet dataSet = loadTestData(config);
        System.out.println(dataSet.getMetaInfo());
        System.out.println("accuracy on test set = "+Accuracy.accuracy(mlLogisticRegression,dataSet));
        System.out.println("overlap on test set = "+ Overlap.overlap(mlLogisticRegression,dataSet));

    }


}
