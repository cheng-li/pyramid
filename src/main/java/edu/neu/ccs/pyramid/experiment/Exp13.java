package edu.neu.ccs.pyramid.experiment;

import edu.neu.ccs.pyramid.configuration.Config;
import edu.neu.ccs.pyramid.dataset.*;
import edu.neu.ccs.pyramid.eval.Accuracy;
import edu.neu.ccs.pyramid.eval.Overlap;
import edu.neu.ccs.pyramid.eval.PerClassMeasures;
import edu.neu.ccs.pyramid.feature.CategoricalFeatureMapper;
import edu.neu.ccs.pyramid.feature.FeatureMappers;
import edu.neu.ccs.pyramid.feature.NumericalFeatureMapper;
import edu.neu.ccs.pyramid.multilabel_classification.hmlgb.HMLGBConfig;
import edu.neu.ccs.pyramid.multilabel_classification.hmlgb.HMLGBInspector;
import edu.neu.ccs.pyramid.multilabel_classification.hmlgb.HMLGBTrainer;
import edu.neu.ccs.pyramid.multilabel_classification.hmlgb.HMLGradientBoosting;
import org.apache.commons.lang3.time.StopWatch;
import org.apache.mahout.math.Vector;

import java.io.File;
import java.util.Arrays;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import java.util.stream.Collectors;


/**
 * hmlgb
 * Created by chengli on 10/11/14.
 */
public class Exp13 {
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


        Set<String> featuresToUseOption = Arrays.stream(config.getString("train.features").split(",")).map(string -> string.trim())
                .collect(Collectors.toSet());
        FeatureMappers featureMappers = dataSet.getSetting().getFeatureMappers();

        Set<Integer> initialFeatures = new HashSet<>();
        for (CategoricalFeatureMapper mapper: featureMappers.getCategoricalFeatureMappers()){
            if (mapper.getSettings().get("source").equalsIgnoreCase("field")){
                for (int j = mapper.getStart();j<=mapper.getEnd();j++){
                    initialFeatures.add(j);
                }
            }
        }
        for (NumericalFeatureMapper mapper: featureMappers.getNumericalFeatureMappers()){
            if (mapper.getSettings().get("source").equalsIgnoreCase("field")){
                initialFeatures.add(mapper.getFeatureIndex());
            }
        }

        Set<Integer> unigramFeatures = new HashSet<>();
        for (NumericalFeatureMapper mapper: featureMappers.getNumericalFeatureMappers()){
            if (mapper.getSettings().get("source").equalsIgnoreCase("matching_score")
                    && mapper.getSettings().get("ngram").split(" ").length==1){
                unigramFeatures.add(mapper.getFeatureIndex());
            }
        }

        Set<Integer> ngramFeatures = new HashSet<>();
        for (NumericalFeatureMapper mapper: featureMappers.getNumericalFeatureMappers()){
            if (mapper.getSettings().get("source").equalsIgnoreCase("matching_score")
                    && mapper.getSettings().get("ngram").split(" ").length>1){
                ngramFeatures.add(mapper.getFeatureIndex());
            }
        }

        if (initialFeatures.size()+unigramFeatures.size()+ngramFeatures.size()!=dataSet.getNumFeatures()){
            throw new RuntimeException("initialFeatures.size()+unigramFeatures.size()+ngramFeatures.size()!=dataSet.getNumFeatures()");
        }

        Set<Integer> featuresToUse = new HashSet<>();
        if (featuresToUseOption.contains("initial")){
            featuresToUse.addAll(initialFeatures);
        }
        if (featuresToUseOption.contains("unigram")){
            featuresToUse.addAll(unigramFeatures);
        }
        if (featuresToUseOption.contains("ngram")){
            featuresToUse.addAll(ngramFeatures);
        }

        int[] activeFeatures = featuresToUse.stream().mapToInt(i-> i).toArray();

        int numClasses = dataSet.getNumClasses();
        System.out.println("number of class = "+numClasses);
        HMLGBConfig hmlgbConfig = new HMLGBConfig.Builder(dataSet)
                .dataSamplingRate(dataSamplingRate)
                .featureSamplingRate(featureSamplingRate)
                .learningRate(learningRate)
                .minDataPerLeaf(minDataPerLeaf)
                .numLeaves(numLeaves)
                .numSplitIntervals(config.getInt("train.numSplitIntervals"))
                .build();

        List<MultiLabel> legalAssignments = DataSetUtil.gatherLabels(dataSet).stream()
                .collect(Collectors.toList());

        HMLGradientBoosting boosting;
        if (config.getBoolean("train.warmStart")){
            boosting = HMLGradientBoosting.deserialize(new File(archive,modelName));
        } else {
            boosting = new HMLGradientBoosting(numClasses,legalAssignments);
        }

        HMLGBTrainer trainer = new HMLGBTrainer(hmlgbConfig,boosting);

        //todo make it better
        trainer.setActiveFeatures(activeFeatures);

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
        File serializedModel =  new File(archive,modelName);


        boosting.serialize(serializedModel);
        System.out.println(stopWatch);

    }

    static void verify(Config config) throws Exception{
        String archive = config.getString("archive.folder");
        String modelName = config.getString("archive.model");

        HMLGradientBoosting boosting = HMLGradientBoosting.deserialize(new File(archive,modelName));
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
                List<String> featureNames = HMLGBInspector.topFeatureNames(boosting, k);
                System.out.println("top features for class " + k + "(" + labelTranslator.toExtLabel(k) + "):");
                System.out.println(featureNames);
            }
        }

        if (config.getBoolean("verify.topNgramsFeatures")){
            for (int k=0;k<dataSet.getNumClasses();k++) {
                List<String> featureNames = HMLGBInspector.topFeatureNames(boosting, k)
                        .stream().filter(name -> name.split(" ").length>1)
                        .collect(Collectors.toList());
                System.out.println("top ngram features for class " + k + "(" + labelTranslator.toExtLabel(k) + "):");
                System.out.println(featureNames);
            }
        }

        if (config.getBoolean("verify.analyzeMistakes")){
            System.out.println("analyzing mistakes");
            analyzeTrainMistakes(config, boosting, dataSet);
        }

    }

    static void test(Config config) throws Exception{
        String archive = config.getString("archive.folder");
        String modelName = config.getString("archive.model");

        HMLGradientBoosting boosting = HMLGradientBoosting.deserialize(new File(archive,modelName));
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
            analyzeTestMistakes(config, boosting, dataSet);
        }

    }

    static void analyzeTrainMistakes(Config config, HMLGradientBoosting boosting, MultiLabelClfDataSet dataSet) throws Exception{
        int numClasses = dataSet.getNumClasses();
        LabelTranslator labelTranslator = dataSet.getSetting().getLabelTranslator();

        List<MultiLabel> predictions = boosting.predict(dataSet);
        MultiLabel[] trueLabels = dataSet.getMultiLabels();

        for (int k=0;k<numClasses;k++){
            PerClassMeasures perClassMeasures = new PerClassMeasures(trueLabels,predictions,k,labelTranslator.toExtLabel(k));
            System.out.println("train: " + perClassMeasures);
        }

        int limit = config.getInt("verify.analyzeMistakes.limit");
        for (int i=0;i<dataSet.getNumDataPoints();i++){

            MultiLabel prediction = predictions.get(i);
            MultiLabel trueLabel = trueLabels[i];
            if (!prediction.equals(trueLabel)){
                System.out.println("=======================================");
                Vector vector = dataSet.getRow(i);
                System.out.println("data point "+i+" index id = "+dataSet.getDataPointSetting(i).getExtId());

                System.out.println(HMLGBInspector.analyzeMistake(boosting,vector,trueLabel,prediction,labelTranslator,limit));


            }
        }
    }

    static void analyzeTestMistakes(Config config, HMLGradientBoosting boosting, MultiLabelClfDataSet dataSet) throws Exception{
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

        for (int k=0;k<numClassesInTest;k++){
            PerClassMeasures perClassMeasures = new PerClassMeasures(trueLabels,predictions,k,labelTranslator.toExtLabel(k));
            System.out.println("test: " + perClassMeasures);
        }

        int limit = config.getInt("test.analyzeMistakes.limit");
        for (int i=0;i<dataSet.getNumDataPoints();i++){

            MultiLabel prediction = predictions.get(i);
            MultiLabel trueLabel = trueLabels[i];
            if (!prediction.equals(trueLabel)){
                System.out.println("=======================================");
                Vector vector = dataSet.getRow(i);
                System.out.println("data point "+i+" index id = "+dataSet.getDataPointSetting(i).getExtId());
                if (trueLabel.outOfBound(numClassesInTrain)){
                    System.out.println("true labels = "+trueLabel.toStringWithExtLabels(labelTranslator));
                    System.out.println("predicted labels = "+prediction.toStringWithExtLabels(labelTranslator));
                    System.out.println("it contains unseen labels");
                } else{
                    System.out.println(HMLGBInspector.analyzeMistake(boosting,vector,trueLabel,prediction,labelTranslator,limit));
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
