package edu.neu.ccs.pyramid.experiment;

import edu.neu.ccs.pyramid.classification.boosting.lktb.LKTBInspector;
import edu.neu.ccs.pyramid.configuration.Config;
import edu.neu.ccs.pyramid.dataset.*;
import edu.neu.ccs.pyramid.eval.Accuracy;
import edu.neu.ccs.pyramid.eval.MacroAveragedMeasures;
import edu.neu.ccs.pyramid.eval.Overlap;
import edu.neu.ccs.pyramid.multilabel_classification.imlgb.IMLGBConfig;
import edu.neu.ccs.pyramid.multilabel_classification.imlgb.IMLGBInspector;
import edu.neu.ccs.pyramid.multilabel_classification.imlgb.IMLGradientBoosting;
import org.apache.commons.lang3.time.StopWatch;

import java.io.File;
import java.util.List;
import java.util.Set;
import java.util.stream.Collectors;

/**
 * imlgb
 * Created by chengli on 10/11/14.
 */
public class Exp14 {
    public static void main(String[] args) throws Exception{
        if (args.length !=1){
            throw new IllegalArgumentException("please specify the config file");
        }

        Config config = new Config(args[0]);
        System.out.println(config);

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

        if (config.getBoolean("featureMatrix.sparse")){
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

        if (config.getBoolean("featureMatrix.sparse")){
            dataSet= TRECFormat.loadMultiLabelClfDataSet(new File(trainFile), DataSetType.ML_CLF_SPARSE,
                    true);
        } else {
            dataSet= TRECFormat.loadMultiLabelClfDataSet(new File(trainFile), DataSetType.ML_CLF_DENSE,
                    true);
        }

        return dataSet;
    }

    static void train(Config config) throws Exception{
        String archive = config.getString("archive.folder");
        int numIterations = config.getInt("train.numIterations");
        int numLeaves = config.getInt("train.numLeaves");
        double learningRate = config.getDouble("train.learningRate");
        int minDataPerLeaf = config.getInt("train.minDataPerLeaf");
        String modelName = config.getString("archive.model");
        double featureSamplingRate = config.getDouble("train.featureSamplingRate");
        double dataSamplingRate = config.getDouble("train.dataSamplingRate");

        StopWatch stopWatch = new StopWatch();
        stopWatch.start();

        MultiLabelClfDataSet dataSet = loadTrainData(config);
        MultiLabelClfDataSet testDataSet = loadTestData(config);

        int numClasses = dataSet.getNumClasses();
        System.out.println("number of class = "+numClasses);
        IMLGBConfig imlgbConfig = new IMLGBConfig.Builder(dataSet)
                .dataSamplingRate(dataSamplingRate)
                .featureSamplingRate(featureSamplingRate)
                .learningRate(learningRate)
                .minDataPerLeaf(minDataPerLeaf)
                .numLeaves(numLeaves)
                .numSplitIntervals(config.getInt("train.numSplitIntervals"))
                .build();

        IMLGradientBoosting boosting = new IMLGradientBoosting(numClasses);
        List<MultiLabel> assignments = DataSetUtil.gatherLabels(dataSet);
        boosting.setAssignments(assignments);
        boosting.setPriorProbs(dataSet);
        boosting.setTrainConfig(imlgbConfig);

        for (int i=0;i<numIterations;i++){
            System.out.println("iteration "+i);
            boosting.boostOneRound();
            if (config.getBoolean("train.showPerformanceEachRound")){
                System.out.println("accuracy on training set = "+ Accuracy.accuracy(boosting,
                        dataSet));
                System.out.println("overlap on training set = "+ Overlap.overlap(boosting,dataSet));

                System.out.println("accuracy on test set = "+ Accuracy.accuracy(boosting,
                        testDataSet));
                System.out.println("overlap on test set = "+ Overlap.overlap(boosting,testDataSet));
            }

        }
        File serializedModel =  new File(archive,modelName);


        boosting.serialize(serializedModel);
        System.out.println(stopWatch);

    }

    static void verify(Config config) throws Exception{
        String archive = config.getString("archive.folder");
        String modelName = config.getString("archive.model");

        IMLGradientBoosting boosting = IMLGradientBoosting.deserialize(new File(archive,modelName));
        MultiLabelClfDataSet dataSet = loadTrainData(config);


        LabelTranslator labelTranslator = dataSet.getSetting().getLabelTranslator();
        System.out.println("accuracy on training set = "+Accuracy.accuracy(boosting,dataSet));
//        System.out.println("overlap on training set = "+ Overlap.overlap(boosting,dataSet));
//        System.out.println("macro-averaged measure on training set:");
//        System.out.println(new MacroAveragedMeasures(boosting,dataSet));
        if (config.getBoolean("verify.showPredictions")){
            List<MultiLabel> prediction = boosting.predict(dataSet);
            for (int i=0;i<dataSet.getNumDataPoints();i++){
                System.out.println(""+i);
                System.out.println("true labels:");
                System.out.println(dataSet.getMultiLabels()[i]);
                StringBuilder trueExtLabels = new StringBuilder();
                for (int matched: dataSet.getMultiLabels()[i].getMatchedLabels()){
                    trueExtLabels.append(labelTranslator.toExtLabel(matched));
                    trueExtLabels.append(", ");
                }
                System.out.println(trueExtLabels);
                System.out.println("predictions:");
                System.out.println(prediction.get(i));
                StringBuilder predictedExtLabels = new StringBuilder();
                for (int matched: prediction.get(i).getMatchedLabels()){
                    predictedExtLabels.append(labelTranslator.toExtLabel(matched));
                    predictedExtLabels.append(", ");
                }
                System.out.println(predictedExtLabels);
            }
        }
        if (config.getBoolean("verify.topFeatures")){

            for (int k=0;k<dataSet.getNumClasses();k++) {
                List<String> featureNames = IMLGBInspector.topFeatureNames(boosting, k);
                System.out.println("top features for class " + k + "(" + labelTranslator.toExtLabel(k) + "):");
                System.out.println(featureNames);
            }
        }

        if (config.getBoolean("verify.topNgramsFeatures")){
            for (int k=0;k<dataSet.getNumClasses();k++) {
                List<String> featureNames = IMLGBInspector.topFeatureNames(boosting, k)
                        .stream().filter(name -> name.split(" ").length>1)
                        .collect(Collectors.toList());
                System.out.println("top ngram features for class " + k + "(" + labelTranslator.toExtLabel(k) + "):");
                System.out.println(featureNames);
            }
        }

    }

    static void test(Config config) throws Exception{
        String archive = config.getString("archive.folder");
        String modelName = config.getString("archive.model");

        IMLGradientBoosting boosting = IMLGradientBoosting.deserialize(new File(archive,modelName));
        MultiLabelClfDataSet dataSet = loadTestData(config);
        System.out.println("accuracy on test set = "+Accuracy.accuracy(boosting,dataSet));
        System.out.println("overlap on test set = "+ Overlap.overlap(boosting,dataSet));
//        System.out.println("macro-averaged measure on test set:");
//        System.out.println(new MacroAveragedMeasures(boosting,dataSet));
        if (config.getBoolean("test.showPredictions")){
            List<MultiLabel> prediction = boosting.predict(dataSet);
            for (int i=0;i<dataSet.getNumDataPoints();i++){
                System.out.println(""+i);
                System.out.println("true labels:");
                System.out.println(dataSet.getMultiLabels()[i]);
                System.out.println("predictions:");
                System.out.println(prediction.get(i));
            }
        }

        if (config.getBoolean("test.analyzeMistakes")){
            System.out.println("analyzing mistakes");
            analyzeMistakes(config,boosting,dataSet);
        }

    }


    static void analyzeMistakes(Config config, IMLGradientBoosting boosting, MultiLabelClfDataSet dataSet) throws Exception{
        int numClassesInTrain = getNumClassesInTrain(config);
        int numClassesInTest = dataSet.getNumClasses();
        LabelTranslator labelTranslator = dataSet.getSetting().getLabelTranslator();
        if (numClassesInTest>numClassesInTrain){
            System.out.println("new labels not seen in the training set:");
            for (int k=numClassesInTrain;k<numClassesInTest;k++){
                System.out.println(""+k+"("+labelTranslator.toExtLabel(k)+")");
            }
        }
        List<MultiLabel> predictions = boosting.predict(dataSet);
        MultiLabel[] trueLabels = dataSet.getMultiLabels();

        int limit = config.getInt("test.analyzeMistakes.limit");
        for (int i=0;i<dataSet.getNumDataPoints();i++){
            System.out.println("=======================================");
            MultiLabel prediction = predictions.get(i);
            MultiLabel trueLabel = trueLabels[i];
            if (!prediction.equals(trueLabel)){
                FeatureRow featureRow = dataSet.getFeatureRow(i);
                System.out.println("data point "+i+" index id = "+featureRow.getSetting().getExtId());
                if (prediction.outOfBound(numClassesInTrain)){
                    System.out.println("true labels = "+trueLabel.toStringWithExtLabels(labelTranslator));
                    System.out.println("predicted labels = "+prediction.toStringWithExtLabels(labelTranslator));
                    System.out.println("it contains unseen labels");
                } else{
                    System.out.println(IMLGBInspector.analyzeMistake(boosting,featureRow,trueLabel,prediction,labelTranslator,limit));
                }

            }
        }
    }

    static int getNumClassesInTrain(Config config) throws Exception{
        MultiLabelClfDataSet dataSet = loadTrainData(config);
        int numClasses = dataSet.getNumClasses();
        return numClasses;
    }


}
