package edu.neu.ccs.pyramid.experiment;

import edu.neu.ccs.pyramid.configuration.Config;
import edu.neu.ccs.pyramid.dataset.DataSetType;
import edu.neu.ccs.pyramid.dataset.MultiLabel;
import edu.neu.ccs.pyramid.dataset.MultiLabelClfDataSet;
import edu.neu.ccs.pyramid.dataset.TRECFormat;
import edu.neu.ccs.pyramid.eval.Accuracy;
import edu.neu.ccs.pyramid.eval.Overlap;
import edu.neu.ccs.pyramid.multilabel_classification.bmm_variant.BMMClassifier;
import edu.neu.ccs.pyramid.util.Pair;
import edu.neu.ccs.pyramid.util.SetUtil;

import java.io.File;
import java.io.IOException;
import java.nio.file.Paths;
import java.util.HashSet;
import java.util.Set;
import java.util.regex.Pattern;

/**
 * Created by Rainicy on 1/23/16.
 */
public class Exp225 {

    public static void main(String[] args) throws Exception {
        if (args.length != 1) {
            throw new IllegalArgumentException("Please specify a properties file.");
        }

        Config config = new Config(args[0]);

        System.out.println(config);

        String matrixType = config.getString("input.matrixType");

        MultiLabelClfDataSet trainSet;
        MultiLabelClfDataSet testSet;

        switch (matrixType){
            case "sparse_random":
                trainSet= TRECFormat.loadMultiLabelClfDataSet(config.getString("input.trainData"),
                        DataSetType.ML_CLF_SPARSE, true);
                testSet = TRECFormat.loadMultiLabelClfDataSet(config.getString("input.testData"),
                        DataSetType.ML_CLF_SPARSE, true);
                break;
            case "sparse_sequential":
                trainSet= TRECFormat.loadMultiLabelClfDataSet(config.getString("input.trainData"),
                        DataSetType.ML_CLF_SEQ_SPARSE, true);
                testSet = TRECFormat.loadMultiLabelClfDataSet(config.getString("input.testData"),
                        DataSetType.ML_CLF_SEQ_SPARSE, true);
                break;
            case "dense":
                trainSet= TRECFormat.loadMultiLabelClfDataSet(config.getString("input.trainData"),
                        DataSetType.ML_CLF_DENSE, true);
                testSet = TRECFormat.loadMultiLabelClfDataSet(config.getString("input.testData"),
                        DataSetType.ML_CLF_DENSE, true);
                break;
            default:
                throw new IllegalArgumentException("unknown type");
        }

        String output = config.getString("output");
        String modelName = config.getString("modelName");
        File path = Paths.get(output, modelName).toFile();
        path.mkdirs();

        BMMClassifier bmmClassifier = BMMClassifier.deserialize(new File(path, "model"));
        bmmClassifier.setAllowEmpty(config.getBoolean("predict.allowEmpty"));


        System.out.println("--------------------------------Analysis-----------------------------\n");
        Set<MultiLabel> trainLabelSet = new HashSet<>();
        Set<MultiLabel> testLabelSet  = new HashSet<>();

        for (MultiLabel multiLabel : trainSet.getMultiLabels()) {
            trainLabelSet.add(multiLabel);
        }
        for (MultiLabel multiLabel : testSet.getMultiLabels()) {
            testLabelSet.add(multiLabel);
        }

        Set<MultiLabel> newTestSet = SetUtil.complement(testLabelSet, trainLabelSet);
        System.out.println("New label combination number in test: " + newTestSet.size());
        int newTestLabelCounts = 0;
        for (MultiLabel label : testSet.getMultiLabels()) {
            if (newTestSet.contains(label)) {
                newTestLabelCounts++;
            }
        }
        System.out.println("New label combination data counts: " + newTestLabelCounts);
        System.out.println("New label combination data rate: " + (double)newTestLabelCounts/testSet.getNumDataPoints());


        MultiLabel[] trainPred;
        MultiLabel[] testPred;
        System.out.println("--------------------------------Analysis Basic:dynamic-----------------------------\n");
        bmmClassifier.setPredictMode("dynamic");
//        bmmClassifier.setNumSample(config.getInt("predict.sampling.numSamples"));
        trainPred = bmmClassifier.predict(trainSet);
        testPred = bmmClassifier.predict(testSet);
        int newPredTrueCount = 0;
        int newPredFalseCount = 0;
        for (int i=0; i<testPred.length; i++) {
            MultiLabel label = testPred[i];
            if (!trainLabelSet.contains(label) && label.equals(testSet.getMultiLabels()[i])) {
                newPredTrueCount++;
            }
            if (!trainLabelSet.contains(label) && !label.equals(testSet.getMultiLabels()[i])) {
                newPredFalseCount++;
            }
        }
        int totalCount = newPredFalseCount + newPredTrueCount;
        System.out.println("New label prediction data counts: " + totalCount);
        System.out.println("New label prediction true data: " + newPredTrueCount + "#\t" + (double)newPredTrueCount/testSet.getNumDataPoints());
        System.out.println("New label prediction false data: " + newPredFalseCount + "#\t" + (double)newPredFalseCount/testSet.getNumDataPoints());
        System.out.println();
        System.out.print("trainAcc : " + Accuracy.accuracy(trainSet.getMultiLabels(), trainPred) + "\t");
        System.out.print("trainOver: "+ Overlap.overlap(trainSet.getMultiLabels(), trainPred)+ "\t");
        System.out.print("testAcc  : "+ Accuracy.accuracy(testSet.getMultiLabels(),testPred)+ "\t");
        System.out.println("testOver : "+ Overlap.overlap(testSet.getMultiLabels(), testPred)+ "\t");
        System.out.println();
        System.out.println();
        System.out.println("--------------------------------Analysis Basic:sampling-----------------------------\n");
        bmmClassifier.setPredictMode("sampling");
        bmmClassifier.setNumSample(config.getInt("predict.sampling.numSamples"));
        trainPred = bmmClassifier.predict(trainSet);
        testPred = bmmClassifier.predict(testSet);
        newPredTrueCount = 0;
        newPredFalseCount = 0;
        for (int i=0; i<testPred.length; i++) {
            MultiLabel label = testPred[i];
            if (!trainLabelSet.contains(label) && label.equals(testSet.getMultiLabels()[i])) {
                newPredTrueCount++;
            }
            if (!trainLabelSet.contains(label) && !label.equals(testSet.getMultiLabels()[i])) {
                newPredFalseCount++;
            }
        }
        totalCount = newPredFalseCount + newPredTrueCount;
        System.out.println("New label prediction data counts: " + totalCount);
        System.out.println("New label prediction true data: " + newPredTrueCount + "#\t" + (double)newPredTrueCount/testSet.getNumDataPoints());
        System.out.println("New label prediction false data: " + newPredFalseCount + "#\t" + (double)newPredFalseCount/testSet.getNumDataPoints());
        System.out.println();
        System.out.print("trainAcc : " + Accuracy.accuracy(trainSet.getMultiLabels(), trainPred) + "\t");
        System.out.print("trainOver: "+ Overlap.overlap(trainSet.getMultiLabels(), trainPred)+ "\t");
        System.out.print("testAcc  : "+ Accuracy.accuracy(testSet.getMultiLabels(),testPred)+ "\t");
        System.out.println("testOver : "+ Overlap.overlap(testSet.getMultiLabels(), testPred)+ "\t");
        System.out.println();
        System.out.println();


        System.out.println("--------------------------------Analysis Given Train Labels-----------------------------\n");
        trainPred = bmmClassifier.predict(trainSet, trainSet.getMultiLabels());
        testPred = bmmClassifier.predict(testSet, trainSet.getMultiLabels());
        System.out.println();
        System.out.print("trainAcc : " + Accuracy.accuracy(trainSet.getMultiLabels(), trainPred) + "\t");
        System.out.print("trainOver: "+ Overlap.overlap(trainSet.getMultiLabels(), trainPred)+ "\t");
        System.out.print("testAcc  : "+ Accuracy.accuracy(testSet.getMultiLabels(),testPred)+ "\t");
        System.out.println("testOver : "+ Overlap.overlap(testSet.getMultiLabels(), testPred)+ "\t");
        System.out.println();
        System.out.println();
        System.out.println("--------------------------------Analysis Given Test Labels-----------------------------\n");
        trainPred = bmmClassifier.predict(trainSet, testSet.getMultiLabels());
        testPred = bmmClassifier.predict(testSet, testSet.getMultiLabels());
        System.out.println();
        System.out.print("trainAcc : " + Accuracy.accuracy(trainSet.getMultiLabels(), trainPred) + "\t");
        System.out.print("trainOver: "+ Overlap.overlap(trainSet.getMultiLabels(), trainPred)+ "\t");
        System.out.print("testAcc  : "+ Accuracy.accuracy(testSet.getMultiLabels(),testPred)+ "\t");
        System.out.println("testOver : "+ Overlap.overlap(testSet.getMultiLabels(), testPred)+ "\t");
        System.out.println();
        System.out.println();
    }

}


