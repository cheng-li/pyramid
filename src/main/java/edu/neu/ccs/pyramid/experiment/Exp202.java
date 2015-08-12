package edu.neu.ccs.pyramid.experiment;

import edu.neu.ccs.pyramid.configuration.Config;
import edu.neu.ccs.pyramid.data_formatter.reuters.IndexBuilder;
import edu.neu.ccs.pyramid.dataset.*;
import edu.neu.ccs.pyramid.eval.*;
import edu.neu.ccs.pyramid.multilabel_classification.imlgb.IMLGradientBoosting;
import edu.neu.ccs.pyramid.util.Serialization;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Created by Rainicy on 6/27/15.
 */
public class Exp202 {

    public static void main(String[] args) throws Exception {
        if (args.length != 1) {
            throw new IllegalArgumentException("please specify the conifg file.");
        }

        Config config = new Config(args[0]);

        String path = config.getString("path");
        String codesDictPath = config.getString("codes.dict");
        Map<String, String> codesDictMap = IndexBuilder.getCodesDict(codesDictPath);


        List<String> labels = new ArrayList<>();
        for(Map.Entry<String, String> entry : codesDictMap.entrySet()) {
            String code = entry.getKey();
            String desc = entry.getValue();
            labels.add(code + " : " + desc);
        }

        // deserialize
        IMLGradientBoosting imlGradientBoosting = (IMLGradientBoosting) Serialization.deserialize(new File(path,"model"));
        // load data
        MultiLabelClfDataSet train = TRECFormat.loadMultiLabelClfDataSet(new File(new File(path, "data_sets"), "train"), DataSetType.ML_CLF_SPARSE ,true);
        MultiLabelClfDataSet test = TRECFormat.loadMultiLabelClfDataSet(new File(new File(path, "data_sets"), "test"), DataSetType.ML_CLF_SPARSE ,true);

        MultiLabel[] testTrue = test.getMultiLabels();
        MultiLabel[] trainTrue = train.getMultiLabels();
        MultiLabel[] testPredict = imlGradientBoosting.predict(test);
        MultiLabel[] trainPredict = imlGradientBoosting.predict(train);

        LabelTranslator labelTranslator = test.getLabelTranslator();


        Map<String, Double> trainAcc = new HashMap<>();
        Map<String, Double> testAcc = new HashMap<>();
        Map<String, Double> trainPrecision = new HashMap<>();
        Map<String, Double> testPrecision = new HashMap<>();
        Map<String, Double> trainRecall = new HashMap<>();
        Map<String, Double> testRecall = new HashMap<>();
        Map<String, Double> trainF1 = new HashMap<>();
        Map<String, Double> testF1 = new HashMap<>();

//        int labelId = 0;
        for (String careLabel : labels) {
//            System.out.println(++labelId + ": " + careLabel);
            int careLabelIndex = 0;
            try {
                careLabelIndex = labelTranslator.toIntLabel(careLabel);
            } catch (Exception e) {
                continue;
            }


            int[] trueLabelsTrain = new int[trainTrue.length];
            int[] predictionTrain = new int[trainTrue.length];
            // get trueLabels and prediction arrays for train
            for (int i=0; i<trainTrue.length; i++) {
                MultiLabel yHat = trainPredict[i];
                MultiLabel y = trainTrue[i];

                if (yHat.matchClass(careLabelIndex)) {
                    predictionTrain[i] = 1;
                } else {
                    predictionTrain[i] = 0;
                }
                if (y.matchClass(careLabelIndex)) {
                    trueLabelsTrain[i] = 1;
                } else {
                    trueLabelsTrain[i] = 0;
                }
            }
            trainAcc.put(careLabel, Accuracy.accuracy(trueLabelsTrain, predictionTrain));
            trainPrecision.put(careLabel, Precision.precision(trueLabelsTrain, predictionTrain, 1));
            trainRecall.put(careLabel, Recall.recall(trueLabelsTrain, predictionTrain, 1));
            trainF1.put(careLabel, FMeasure.f1(trainPrecision.get(careLabel), trainRecall.get(careLabel)));

            int[] trueLabelsTest = new int[testTrue.length];
            int[] predictionTest = new int[testTrue.length];
            // get trueLabels and prediction arrays for train
            for (int i=0; i<testTrue.length; i++) {
                MultiLabel yHat = testPredict[i];
                MultiLabel y = testTrue[i];

                if (yHat.matchClass(careLabelIndex)) {
                    predictionTest[i] = 1;
                } else {
                    predictionTest[i] = 0;
                }
                if (y.matchClass(careLabelIndex)) {
                    trueLabelsTest[i] = 1;
                } else {
                    trueLabelsTest[i] = 0;
                }
            }
            testAcc.put(careLabel, Accuracy.accuracy(trueLabelsTest, predictionTest));
            testPrecision.put(careLabel, Precision.precision(trueLabelsTest, predictionTest, 1));
            testRecall.put(careLabel, Recall.recall(trueLabelsTest, predictionTest, 1));
            testF1.put(careLabel, FMeasure.f1(testPrecision.get(careLabel), testRecall.get(careLabel)));

        }

        String output = config.getString("output");
        String outputTrain = output + ".train.txt";
        String outputTest = output + ".test.txt";
        BufferedWriter bw = new BufferedWriter(new FileWriter(outputTrain));
        bw.write("accuracy on training set = " + Accuracy.accuracy(trainTrue, trainPredict) + "\n");
        bw.write("overlap on training set = " + Overlap.overlap(trainTrue, trainPredict) + "\n");
        bw.write("Label\tAccuracy\tPrecision\tRecall\tF1\n");
        for (Map.Entry<String, Double> entry : trainAcc.entrySet()) {
            String keyLabel = entry.getKey();
            bw.write(keyLabel + "\t" + trainAcc.get(keyLabel) + "\t" + trainPrecision.get(keyLabel) + "\t" +
            trainRecall.get(keyLabel) + "\t" + trainF1.get(keyLabel) + "\n");
        }
        bw.close();

        bw = new BufferedWriter(new FileWriter(outputTest));
        bw.write("accuracy on testing set = " + Accuracy.accuracy(testTrue, testPredict) + "\n");
        bw.write("overlap on testing set = " + Overlap.overlap(testTrue, testPredict) + "\n");
        bw.write("Label\tAccuracy\tPrecision\tRecall\tF1\n");
        for (Map.Entry<String, Double> entry : testAcc.entrySet()) {
            String keyLabel = entry.getKey();
            bw.write(keyLabel + "\t" + testAcc.get(keyLabel) + "\t" + testPrecision.get(keyLabel) + "\t" +
                    testRecall.get(keyLabel) + "\t" + testF1.get(keyLabel) + "\n");
        }
        bw.close();
    }
}
