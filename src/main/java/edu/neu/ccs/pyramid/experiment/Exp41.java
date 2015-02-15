package edu.neu.ccs.pyramid.experiment;

import edu.neu.ccs.pyramid.classification.logistic_regression.LogisticRegression;
import edu.neu.ccs.pyramid.classification.logistic_regression.LogisticRegressionInspector;
import edu.neu.ccs.pyramid.configuration.Config;
import edu.neu.ccs.pyramid.dataset.*;
import edu.neu.ccs.pyramid.eval.Accuracy;
import edu.neu.ccs.pyramid.eval.Overlap;
import edu.neu.ccs.pyramid.eval.PerClassMeasures;
import edu.neu.ccs.pyramid.feature.FeatureUtility;
import edu.neu.ccs.pyramid.multilabel_classification.multi_label_logistic_regression.MLLogisticLoss;
import edu.neu.ccs.pyramid.multilabel_classification.multi_label_logistic_regression.MLLogisticRegression;
import edu.neu.ccs.pyramid.multilabel_classification.multi_label_logistic_regression.MLLogisticRegressionInspector;
import edu.neu.ccs.pyramid.multilabel_classification.multi_label_logistic_regression.MLLogisticTrainer;
import edu.neu.ccs.pyramid.optimization.LBFGS;
import org.apache.commons.lang3.time.StopWatch;
import org.apache.mahout.math.Vector;

import java.io.File;
import java.util.*;
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
        String archive = config.getString("archive.folder");
        String modelName = config.getString("archive.model");

        StopWatch stopWatch = new StopWatch();
        stopWatch.start();

        MultiLabelClfDataSet dataSet = loadTrainData(config);
        System.out.println("training data set loaded");
        MultiLabelClfDataSet testSet = loadTestData(config);
        System.out.println("test data set loaded");

        System.out.println(dataSet.getMetaInfo());
        System.out.println("gathering assignments ");
        List<MultiLabel> assignments = DataSetUtil.gatherLabels(dataSet).stream()
                .collect(Collectors.toList());
        System.out.println("there are "+assignments.size()+ " assignments");

        System.out.println("initializing logistic regression");
        MLLogisticRegression mlLogisticRegression = new MLLogisticRegression(dataSet.getNumClasses(),dataSet.getNumFeatures(),
                assignments);
        for (int j=0;j<dataSet.getNumFeatures();j++){
            mlLogisticRegression.setFeatureName(j,dataSet.getFeatureSetting(j).getFeatureName());
        }

        mlLogisticRegression.setFeatureExtraction(false);

        System.out.println("done");

        System.out.println("initializing objective function");
        MLLogisticLoss function = new MLLogisticLoss(mlLogisticRegression,dataSet,config.getDouble("train.gaussianPriorVariance"));
        System.out.println("done");
        System.out.println("initializing lbfgs");
        LBFGS lbfgs = new LBFGS(function);
        lbfgs.setEpsilon(config.getDouble("train.epsilon"));
        lbfgs.setHistory(5);
        LinkedList<Double> valueQueue = new LinkedList<>();
        valueQueue.add(function.getValue(function.getParameters()));
        System.out.println("done");

        int iteration=0;
        System.out.println("iteration = "+iteration);
        lbfgs.iterate();
        valueQueue.add(function.getValue(function.getParameters()));
        iteration+=1;
        while(true){
            System.out.println("iteration ="+iteration);
            System.out.println("objective = "+valueQueue.getLast());
            System.out.println("training accuracy = "+Accuracy.accuracy(mlLogisticRegression,dataSet));
            System.out.println("training overlap +"+Overlap.overlap(mlLogisticRegression,dataSet));
            System.out.println("test accuracy = "+Accuracy.accuracy(mlLogisticRegression,testSet));
            System.out.println("test overlap +"+Overlap.overlap(mlLogisticRegression,testSet));
            if (Math.abs(valueQueue.getFirst()-valueQueue.getLast())<config.getDouble("train.epsilon")){
                break;
            }
            lbfgs.iterate();
            valueQueue.remove();
            valueQueue.add(function.getValue(function.getParameters()));
            iteration += 1;
        }
        
//        MLLogisticRegression mlLogisticRegression = trainer.train(dataSet,assignments);
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

    private static void verify(Config config) throws Exception{
        String input = config.getString("input.folder");
        MultiLabelClfDataSet dataSet = TRECFormat.loadMultiLabelClfDataSet(new File(input,"train.trec"),
                DataSetType.ML_CLF_SPARSE, true);
        LabelTranslator labelTranslator = dataSet.getSetting().getLabelTranslator();
        File modelFile = new File(config.getString("archive.folder"),config.getString("archive.model"));
        MLLogisticRegression mlLogisticRegression = MLLogisticRegression.deserialize(modelFile);


        int limit = config.getInt("verify.topFeature.limit");
        for (int k=0;k<mlLogisticRegression.getNumClasses();k++){
            System.out.println("top feature for class "+k+"("+labelTranslator.toExtLabel(k)+")");
            System.out.println(MLLogisticRegressionInspector.topFeatures(mlLogisticRegression, k)
                    .stream().limit(limit).map(FeatureUtility::getName).collect(Collectors.toList()));
        }
    }


}
