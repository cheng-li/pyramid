package edu.neu.ccs.pyramid.experiment;

import com.fasterxml.jackson.databind.ObjectMapper;
import edu.neu.ccs.pyramid.classification.logistic_regression.LogisticRegression;
import edu.neu.ccs.pyramid.classification.logistic_regression.LogisticRegressionInspector;
import edu.neu.ccs.pyramid.configuration.Config;
import edu.neu.ccs.pyramid.dataset.*;
import edu.neu.ccs.pyramid.eval.Accuracy;
import edu.neu.ccs.pyramid.eval.Overlap;
import edu.neu.ccs.pyramid.eval.PerClassMeasures;
import edu.neu.ccs.pyramid.feature.FeatureUtility;
import edu.neu.ccs.pyramid.feature.TopFeatures;
import edu.neu.ccs.pyramid.multilabel_classification.MultiLabelPredictionAnalysis;
import edu.neu.ccs.pyramid.multilabel_classification.imlgb.IMLGBInspector;
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
import java.util.stream.IntStream;

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
        String archive = config.getString("output.folder");
        String modelName = config.getString("output.model");

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
        mlLogisticRegression.setFeatureList(dataSet.getFeatureList());
        mlLogisticRegression.setLabelTranslator(dataSet.getLabelTranslator());

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
        String archive = config.getString("output.folder");
        String modelName = config.getString("output.model");

        MLLogisticRegression mlLogisticRegression = MLLogisticRegression.deserialize(new File(archive,modelName));
        System.out.println("test data set loaded");
        MultiLabelClfDataSet dataSet = loadTestData(config);
        System.out.println(dataSet.getMetaInfo());
        System.out.println("accuracy on test set = "+Accuracy.accuracy(mlLogisticRegression,dataSet));
        System.out.println("overlap on test set = "+ Overlap.overlap(mlLogisticRegression,dataSet));


        if (config.getBoolean("test.analyze")){
            int limit = config.getInt("test.analyze.rule.limit");


            List<MultiLabelPredictionAnalysis> analysisList = IntStream.range(0,dataSet.getNumDataPoints()).parallel().filter(
                    i -> {
                        MultiLabel multiLabel = dataSet.getMultiLabels()[i];
                        MultiLabel prediction = mlLogisticRegression.predict(dataSet.getRow(i));
                        boolean accept = false;
                        if (config.getBoolean("test.analyze.doc.withRightPrediction")) {
                            accept = accept || multiLabel.equals(prediction);
                        }

                        if (config.getBoolean("test.analyze.doc.withWrongPrediction")) {
                            accept = accept || !multiLabel.equals(prediction);
                        }
                        return accept;
                    }
            ).mapToObj(i -> {
                        MultiLabel multiLabel = dataSet.getMultiLabels()[i];
                        MultiLabel prediction = mlLogisticRegression.predict(dataSet.getRow(i));
                        List<Integer> classes = new ArrayList<Integer>();
                        for (int k = 0; k < dataSet.getNumClasses(); k++) {
                            boolean condition1 = multiLabel.matchClass(k) && prediction.matchClass(k) && config.getBoolean("test.analyze.class.truePositive");
                            boolean condition2 = !multiLabel.matchClass(k) && !prediction.matchClass(k) && config.getBoolean("test.analyze.class.trueNegative");
                            boolean condition3 = !multiLabel.matchClass(k) && prediction.matchClass(k) && config.getBoolean("test.analyze.class.falsePositive");
                            boolean condition4 = multiLabel.matchClass(k) && !prediction.matchClass(k) && config.getBoolean("test.analyze.class.falseNegative");
                            boolean condition5 = k<mlLogisticRegression.getNumClasses();
                            boolean accept = (condition1 || condition2 || condition3 || condition4) && condition5;
                            if (accept) {
                                classes.add(k);
                            }
                        }
                        return MLLogisticRegressionInspector.analyzePrediction(mlLogisticRegression, dataSet, i, classes, limit);
                    }
            )
                    .collect(Collectors.toList());

            int numDocsPerFile = config.getInt("test.analyze.numDocsPerFile");
            int numFiles = (int)Math.ceil((double)analysisList.size()/numDocsPerFile);


            for (int i=0;i<numFiles;i++){
                int start = i;
                int end = i+numDocsPerFile;
                List<MultiLabelPredictionAnalysis> partition = new ArrayList<>();
                for (int a=start;a<end && a<analysisList.size();a++){
                    partition.add(analysisList.get(a));
                }
                ObjectMapper mapper = new ObjectMapper();
                String fileName = config.getString("test.analyze.file");
                int suffixIndex = fileName.lastIndexOf(".json");
                if (suffixIndex==-1){
                    suffixIndex=fileName.length();
                }
                String file = fileName.substring(0, suffixIndex)+"_"+(i+1)+".json";
                mapper.writeValue(new File(config.getString("output.folder"),file), partition);
            }

        }

    }

    private static void verify(Config config) throws Exception{
        String input = config.getString("input.folder");
        MultiLabelClfDataSet dataSet = TRECFormat.loadMultiLabelClfDataSet(new File(input,"train.trec"),
                DataSetType.ML_CLF_SPARSE, true);
        File modelFile = new File(config.getString("output.folder"),config.getString("output.model"));
        MLLogisticRegression mlLogisticRegression = MLLogisticRegression.deserialize(modelFile);


        if (config.getBoolean("verify.topFeatures")){
            int limit = config.getInt("verify.topFeatures.limit");
            List<TopFeatures> topFeaturesList = IntStream.range(0, dataSet.getNumClasses())
                    .mapToObj(k -> MLLogisticRegressionInspector.topFeatures(mlLogisticRegression, k, limit))
                    .collect(Collectors.toList());
            ObjectMapper mapper = new ObjectMapper();
            String file = config.getString("verify.topFeatures.file");
            mapper.writeValue(new File(config.getString("output.folder"),file), topFeaturesList);

        }


        if (config.getBoolean("verify.analyze")){
            int limit = config.getInt("verify.analyze.rule.limit");


            List<MultiLabelPredictionAnalysis> analysisList = IntStream.range(0,dataSet.getNumDataPoints()).parallel().filter(
                    i -> {
                        MultiLabel multiLabel = dataSet.getMultiLabels()[i];
                        MultiLabel prediction = mlLogisticRegression.predict(dataSet.getRow(i));
                        boolean accept = false;
                        if (config.getBoolean("verify.analyze.doc.withRightPrediction")) {
                            accept = accept || multiLabel.equals(prediction);
                        }

                        if (config.getBoolean("verify.analyze.doc.withWrongPrediction")) {
                            accept = accept || !multiLabel.equals(prediction);
                        }
                        return accept;
                    }
            ).mapToObj(i -> {
                        MultiLabel multiLabel = dataSet.getMultiLabels()[i];
                        MultiLabel prediction = mlLogisticRegression.predict(dataSet.getRow(i));
                        List<Integer> classes = new ArrayList<Integer>();
                        for (int k = 0; k < dataSet.getNumClasses(); k++) {
                            boolean condition1 = multiLabel.matchClass(k) && prediction.matchClass(k) && config.getBoolean("verify.analyze.class.truePositive");
                            boolean condition2 = !multiLabel.matchClass(k) && !prediction.matchClass(k) && config.getBoolean("verify.analyze.class.trueNegative");
                            boolean condition3 = !multiLabel.matchClass(k) && prediction.matchClass(k) && config.getBoolean("verify.analyze.class.falsePositive");
                            boolean condition4 = multiLabel.matchClass(k) && !prediction.matchClass(k) && config.getBoolean("verify.analyze.class.falseNegative");
                            boolean accept = condition1 || condition2 || condition3 || condition4;
                            if (accept) {
                                classes.add(k);
                            }
                        }
                        return MLLogisticRegressionInspector.analyzePrediction(mlLogisticRegression, dataSet, i, classes, limit);
                    }
            )
                    .collect(Collectors.toList());

            int numDocsPerFile = config.getInt("verify.analyze.numDocsPerFile");
            int numFiles = (int)Math.ceil((double)analysisList.size()/numDocsPerFile);


            for (int i=0;i<numFiles;i++){
                int start = i;
                int end = i+numDocsPerFile;
                List<MultiLabelPredictionAnalysis> partition = new ArrayList<>();
                for (int a=start;a<end && a<analysisList.size();a++){
                    partition.add(analysisList.get(a));
                }
                ObjectMapper mapper = new ObjectMapper();
                String fileName = config.getString("verify.analyze.file");
                int suffixIndex = fileName.lastIndexOf(".json");
                if (suffixIndex==-1){
                    suffixIndex=fileName.length();
                }
                String file = fileName.substring(0,suffixIndex)+"_"+(i+1)+".json";
                mapper.writeValue(new File(config.getString("output.folder"),file), partition);
            }

        }


    }


}
