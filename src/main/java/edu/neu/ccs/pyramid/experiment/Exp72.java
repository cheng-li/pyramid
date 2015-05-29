package edu.neu.ccs.pyramid.experiment;

import com.fasterxml.jackson.databind.ObjectMapper;
import edu.neu.ccs.pyramid.classification.PlattScaling;
import edu.neu.ccs.pyramid.configuration.Config;
import edu.neu.ccs.pyramid.dataset.*;
import edu.neu.ccs.pyramid.eval.Accuracy;
import edu.neu.ccs.pyramid.eval.Overlap;

import edu.neu.ccs.pyramid.feature.TopFeatures;
import edu.neu.ccs.pyramid.multilabel_classification.MLACPlattScaling;
import edu.neu.ccs.pyramid.multilabel_classification.MLPlattScaling;
import edu.neu.ccs.pyramid.multilabel_classification.MultiLabelClassifier;
import edu.neu.ccs.pyramid.multilabel_classification.MultiLabelPredictionAnalysis;
import edu.neu.ccs.pyramid.multilabel_classification.adaboostmh.AdaBoostMH;
import edu.neu.ccs.pyramid.multilabel_classification.adaboostmh.AdaBoostMHInspector;
import edu.neu.ccs.pyramid.multilabel_classification.adaboostmh.AdaBoostMHTrainer;

import edu.neu.ccs.pyramid.multilabel_classification.imlgb.IMLGBInspector;
import edu.neu.ccs.pyramid.multilabel_classification.multi_label_logistic_regression.MLFlatScaling;
import edu.neu.ccs.pyramid.util.Serialization;
import org.apache.commons.lang3.time.StopWatch;

import java.io.File;
import java.util.ArrayList;
import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.IntStream;


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
        if (config.getBoolean("verify")){
            verify(config);
        }
        if (config.getBoolean("test")){
            test(config);
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

        MultiLabelClassifier.ClassProbEstimator scaling = null;
        switch (config.getString("train.scaling")){
            case "eachClass":
                scaling = new MLPlattScaling(dataSet,boosting);
                break;
            case "multiLabel":
                scaling = new MLACPlattScaling(dataSet,boosting);
                break;
            case "flat":
                scaling = new MLFlatScaling(dataSet,boosting);
                break;
            default:
                throw new RuntimeException("unknown scaling fashion");
        }

        Serialization.serialize(scaling,new File(output,config.getString("output.plattScaling")));
        System.out.println(stopWatch);
    }

    static void verify(Config config) throws Exception{
        String output = config.getString("output.folder");
        String modelName = config.getString("output.model");
        AdaBoostMH boosting = AdaBoostMH.deserialize(new File(output,modelName));

        MultiLabelClassifier.ClassProbEstimator plattScaling = (MultiLabelClassifier.ClassProbEstimator)Serialization.deserialize(new File(output,config.getString("output.plattScaling")));
        MultiLabelClfDataSet dataSet = loadTrainData(config);
        System.out.println("accuracy on training set = "+Accuracy.accuracy(boosting,dataSet));

        if (config.getBoolean("verify.topFeatures")){
            int limit = config.getInt("verify.topFeatures.limit");
            List<TopFeatures> topFeaturesList = IntStream.range(0, dataSet.getNumClasses())
                    .mapToObj(k -> AdaBoostMHInspector.topFeatures(boosting, k, limit))
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
                        MultiLabel prediction = boosting.predict(dataSet.getRow(i));
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
                        MultiLabel prediction = boosting.predict(dataSet.getRow(i));
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
                        return AdaBoostMHInspector.analyzePrediction(boosting, plattScaling,dataSet, i, classes, limit);
                    }
            )
                    .collect(Collectors.toList());
            int numDocsPerFile = config.getInt("verify.analyze.numDocsPerFile");
            int numFiles = (int)Math.ceil((double)analysisList.size()/numDocsPerFile);


            for (int i=0;i<numFiles;i++){
                int start = i;
                int end = i+numDocsPerFile;
                List<MultiLabelPredictionAnalysis> partition = new ArrayList<>();
                for (int a=start;a<end;a++){
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

    static void test(Config config) throws Exception{
        String output = config.getString("output.folder");
        String modelName = config.getString("output.model");

        AdaBoostMH boosting = AdaBoostMH.deserialize(new File(output,modelName));
        MultiLabelClassifier.ClassProbEstimator plattScaling = (MultiLabelClassifier.ClassProbEstimator)Serialization.deserialize(new File(output,config.getString("output.plattScaling")));
        MultiLabelClfDataSet dataSet = loadTestData(config);
        System.out.println("accuracy on test set = "+Accuracy.accuracy(boosting,dataSet));
        System.out.println("overlap on test set = "+ Overlap.overlap(boosting,dataSet));

        if (config.getBoolean("test.analyze")){
            int limit = config.getInt("test.analyze.rule.limit");


            List<MultiLabelPredictionAnalysis> analysisList = IntStream.range(0,dataSet.getNumDataPoints()).parallel().filter(
                    i -> {
                        MultiLabel multiLabel = dataSet.getMultiLabels()[i];
                        MultiLabel prediction = boosting.predict(dataSet.getRow(i));
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
                        MultiLabel prediction = boosting.predict(dataSet.getRow(i));
                        List<Integer> classes = new ArrayList<Integer>();
                        for (int k = 0; k < dataSet.getNumClasses(); k++) {
                            boolean condition1 = multiLabel.matchClass(k) && prediction.matchClass(k) && config.getBoolean("test.analyze.class.truePositive");
                            boolean condition2 = !multiLabel.matchClass(k) && !prediction.matchClass(k) && config.getBoolean("test.analyze.class.trueNegative");
                            boolean condition3 = !multiLabel.matchClass(k) && prediction.matchClass(k) && config.getBoolean("test.analyze.class.falsePositive");
                            boolean condition4 = multiLabel.matchClass(k) && !prediction.matchClass(k) && config.getBoolean("test.analyze.class.falseNegative");
                            boolean condition5 = k<boosting.getNumClasses();
                            boolean accept = (condition1 || condition2 || condition3 || condition4) && condition5;
                            if (accept) {
                                classes.add(k);
                            }
                        }
                        return AdaBoostMHInspector.analyzePrediction(boosting, plattScaling, dataSet, i, classes, limit);
                    }
            )
                    .collect(Collectors.toList());

            int numDocsPerFile = config.getInt("test.analyze.numDocsPerFile");
            int numFiles = (int)Math.ceil((double)analysisList.size()/numDocsPerFile);


            for (int i=0;i<numFiles;i++){
                int start = i;
                int end = i+numDocsPerFile;
                List<MultiLabelPredictionAnalysis> partition = new ArrayList<>();
                for (int a=start;a<end;a++){
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
}
