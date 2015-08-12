package edu.neu.ccs.pyramid.experiment;

import edu.neu.ccs.pyramid.configuration.Config;
import edu.neu.ccs.pyramid.dataset.*;
import edu.neu.ccs.pyramid.eval.Accuracy;
import edu.neu.ccs.pyramid.eval.Overlap;
import edu.neu.ccs.pyramid.multilabel_classification.imllr.IMLLogisticLoss;
import edu.neu.ccs.pyramid.multilabel_classification.imllr.IMLLogisticRegression;


import edu.neu.ccs.pyramid.optimization.LBFGS;
import edu.neu.ccs.pyramid.util.Serialization;
import org.apache.commons.lang3.time.StopWatch;

import java.io.File;
import java.util.LinkedList;
import java.util.List;
import java.util.stream.Collectors;

/**
 * multi-label logistic regression, trained independently
 * Created by chengli on 5/15/15.
 */
public class Exp104 {
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

        if (config.getBoolean("verify")){
            verify(config);
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
        String archive = config.getString("output.folder");
        String modelName = config.getString("output.model");

        StopWatch stopWatch = new StopWatch();
        stopWatch.start();

        MultiLabelClfDataSet dataSet = loadTrainData(config);
        System.out.println("training data set loaded");
        System.out.println(dataSet.getMetaInfo());
        MultiLabelClfDataSet testSet = loadTestData(config);
        System.out.println("test data set loaded");
        System.out.println(testSet.getMetaInfo());

        System.out.println("gathering assignments ");
        List<MultiLabel> assignments = DataSetUtil.gatherMultiLabels(dataSet).stream()
                .collect(Collectors.toList());
        System.out.println("there are "+assignments.size()+ " assignments");

        System.out.println("initializing logistic regression");
        IMLLogisticRegression logisticRegression = new IMLLogisticRegression(dataSet.getNumClasses(),dataSet.getNumFeatures(),
                assignments);
        logisticRegression.setFeatureList(dataSet.getFeatureList());
        logisticRegression.setLabelTranslator(dataSet.getLabelTranslator());

        System.out.println("done");

        System.out.println("initializing objective function");
        IMLLogisticLoss function = new IMLLogisticLoss(logisticRegression,dataSet,config.getDouble("train.gaussianPriorVariance"));
        System.out.println("done");
        System.out.println("initializing lbfgs");
        LBFGS lbfgs = new LBFGS(function);
        lbfgs.setEpsilon(config.getDouble("train.epsilon"));
        lbfgs.setHistory(5);
        LinkedList<Double> valueQueue = new LinkedList<>();
        valueQueue.add(function.getValue());
        System.out.println("done");

        int iteration=0;
        System.out.println("iteration = "+iteration);
        lbfgs.iterate();
        valueQueue.add(function.getValue());
        iteration+=1;
        while(true){
            System.out.println("iteration = "+iteration);
            System.out.println("objective = "+valueQueue.getLast());
            System.out.println("training accuracy = "+ Accuracy.accuracy(logisticRegression, dataSet));
            System.out.println("training overlap = "+ Overlap.overlap(logisticRegression, dataSet));
            System.out.println("test accuracy = "+Accuracy.accuracy(logisticRegression,testSet));
            System.out.println("test overlap = "+Overlap.overlap(logisticRegression,testSet));
            if (Math.abs(valueQueue.getFirst()-valueQueue.getLast())<config.getDouble("train.epsilon")){
                break;
            }
            lbfgs.iterate();
            valueQueue.remove();
            valueQueue.add(function.getValue());
            iteration += 1;
        }

//        IMLLogisticRegression logisticRegression = trainer.train(dataSet,assignments);
        File serializedModel =  new File(archive,modelName);
        logisticRegression.serialize(serializedModel);
        System.out.println("time spent = " + stopWatch);
        System.out.println("accuracy on training set = " +Accuracy.accuracy(logisticRegression,dataSet));
        System.out.println("overlap on training set = "+ Overlap.overlap(logisticRegression,dataSet));
    }

    static void test(Config config) throws Exception{
        String archive = config.getString("output.folder");
        String modelName = config.getString("output.model");

        IMLLogisticRegression logisticRegression = (IMLLogisticRegression)Serialization.deserialize(new File(archive, modelName));
        System.out.println("test data set loaded");
        MultiLabelClfDataSet dataSet = loadTestData(config);
        System.out.println(dataSet.getMetaInfo());
        System.out.println("accuracy on test set = "+Accuracy.accuracy(logisticRegression,dataSet));
        System.out.println("overlap on test set = "+ Overlap.overlap(logisticRegression,dataSet));




    }

    private static void verify(Config config) throws Exception {
        String input = config.getString("input.folder");
        MultiLabelClfDataSet dataSet = TRECFormat.loadMultiLabelClfDataSet(new File(input, "train.trec"),
                DataSetType.ML_CLF_SPARSE, true);
        File modelFile = new File(config.getString("output.folder"), config.getString("output.model"));
        IMLLogisticRegression logisticRegression = (IMLLogisticRegression)Serialization.deserialize(modelFile);
    }
}
