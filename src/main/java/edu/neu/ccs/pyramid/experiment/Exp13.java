package edu.neu.ccs.pyramid.experiment;

import com.fasterxml.jackson.databind.ObjectMapper;
import edu.neu.ccs.pyramid.configuration.Config;
import edu.neu.ccs.pyramid.dataset.*;
import edu.neu.ccs.pyramid.eval.Accuracy;
import edu.neu.ccs.pyramid.eval.Overlap;
import edu.neu.ccs.pyramid.eval.PerClassMeasures;
import edu.neu.ccs.pyramid.feature.TopFeatures;
import edu.neu.ccs.pyramid.multilabel_classification.MultiLabelPredictionAnalysis;
import edu.neu.ccs.pyramid.multilabel_classification.hmlgb.HMLGBConfig;
import edu.neu.ccs.pyramid.multilabel_classification.hmlgb.HMLGBInspector;
import edu.neu.ccs.pyramid.multilabel_classification.hmlgb.HMLGBTrainer;
import edu.neu.ccs.pyramid.multilabel_classification.hmlgb.HMLGradientBoosting;
import edu.neu.ccs.pyramid.multilabel_classification.imlgb.IMLGBInspector;
import org.apache.commons.lang3.time.StopWatch;
import org.apache.mahout.math.Vector;

import java.io.File;
import java.util.*;
import java.util.stream.Collectors;
import java.util.stream.IntStream;


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
        int numLeaves = config.getInt("train.numLeaves");
        double learningRate = config.getDouble("train.learningRate");
        int minDataPerLeaf = config.getInt("train.minDataPerLeaf");
        String modelName = config.getString("output.model");
        double featureSamplingRate = config.getDouble("train.featureSamplingRate");
        double dataSamplingRate = config.getDouble("train.dataSamplingRate");

        StopWatch stopWatch = new StopWatch();
        stopWatch.start();

        MultiLabelClfDataSet dataSet = loadTrainData(config);
        MultiLabelClfDataSet testDataSet = loadTestData(config);


//        Set<String> featuresToUseOption = Arrays.stream(config.getString("train.features").split(",")).map(string -> string.trim())
//                .collect(Collectors.toSet());
//        FeatureMappers featureMappers = dataSet.getSetting().getFeatureMappers();
//
//        Set<Integer> initialFeatures = new HashSet<>();
//        for (CategoricalFeatureMapper mapper: featureMappers.getCategoricalFeatureMappers()){
//            if (mapper.getSettings().get("source").equalsIgnoreCase("field")){
//                for (int j = mapper.getStart();j<=mapper.getEnd();j++){
//                    initialFeatures.add(j);
//                }
//            }
//        }
//        for (NumericalFeatureMapper mapper: featureMappers.getNumericalFeatureMappers()){
//            if (mapper.getSettings().get("source").equalsIgnoreCase("field")){
//                initialFeatures.add(mapper.getFeatureIndex());
//            }
//        }
//
//        Set<Integer> unigramFeatures = new HashSet<>();
//        for (NumericalFeatureMapper mapper: featureMappers.getNumericalFeatureMappers()){
//            if (mapper.getSettings().get("source").equalsIgnoreCase("matching_score")
//                    && mapper.getSettings().get("ngram").split(" ").length==1){
//                unigramFeatures.add(mapper.getFeatureIndex());
//            }
//        }
//
//        Set<Integer> ngramFeatures = new HashSet<>();
//        for (NumericalFeatureMapper mapper: featureMappers.getNumericalFeatureMappers()){
//            if (mapper.getSettings().get("source").equalsIgnoreCase("matching_score")
//                    && mapper.getSettings().get("ngram").split(" ").length>1){
//                ngramFeatures.add(mapper.getFeatureIndex());
//            }
//        }
//
//        if (initialFeatures.size()+unigramFeatures.size()+ngramFeatures.size()!=dataSet.getNumFeatures()){
//            throw new RuntimeException("initialFeatures.size()+unigramFeatures.size()+ngramFeatures.size()!=dataSet.getNumFeatures()");
//        }
//
//        Set<Integer> featuresToUse = new HashSet<>();
//        if (featuresToUseOption.contains("initial")){
//            featuresToUse.addAll(initialFeatures);
//        }
//        if (featuresToUseOption.contains("unigram")){
//            featuresToUse.addAll(unigramFeatures);
//        }
//        if (featuresToUseOption.contains("ngram")){
//            featuresToUse.addAll(ngramFeatures);
//        }

        int[] activeFeatures = IntStream.range(0, dataSet.getNumFeatures()).toArray();

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
            boosting = HMLGradientBoosting.deserialize(new File(output,modelName));
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
        File serializedModel =  new File(output,modelName);


        boosting.serialize(serializedModel);
        System.out.println(stopWatch);

    }

    static void verify(Config config) throws Exception{
        String output = config.getString("output.folder");
        String modelName = config.getString("output.model");

        HMLGradientBoosting boosting = HMLGradientBoosting.deserialize(new File(output,modelName));
        MultiLabelClfDataSet dataSet = loadTrainData(config);


        LabelTranslator labelTranslator = dataSet.getLabelTranslator();
        System.out.println("accuracy on training set = "+Accuracy.accuracy(boosting,dataSet));
//        System.out.println("overlap on training set = "+ Overlap.overlap(boosting,dataSet));
//        System.out.println("macro-averaged measure on training set:");
//        System.out.println(new MacroAveragedMeasures(boosting,dataSet));

        if (config.getBoolean("verify.topFeatures")){

            int limit = config.getInt("verify.topFeatures.limit");
            List<TopFeatures> topFeaturesList = IntStream.range(0,dataSet.getNumClasses())
                    .mapToObj(k -> HMLGBInspector.topFeatures(boosting, k, limit))
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
                        return HMLGBInspector.analyzePrediction(boosting, dataSet, i, classes, limit);
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

    static void test(Config config) throws Exception{
        String output = config.getString("output.folder");
        String modelName = config.getString("output.model");

        HMLGradientBoosting boosting = HMLGradientBoosting.deserialize(new File(output,modelName));
        MultiLabelClfDataSet dataSet = loadTestData(config);
        System.out.println("accuracy on test set = "+Accuracy.accuracy(boosting,dataSet));
        System.out.println("overlap on test set = "+ Overlap.overlap(boosting,dataSet));
//        System.out.println("macro-averaged measure on test set:");
//        System.out.println(new MacroAveragedMeasures(boosting,dataSet));

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
                        return HMLGBInspector.analyzePrediction(boosting, dataSet, i, classes, limit);
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



//    static void analyzeTestMistakes(Config config, HMLGradientBoosting boosting, MultiLabelClfDataSet dataSet) throws Exception{
//        int numClassesInTrain = getNumClassesInTrain(config);
//        int numClassesInTest = dataSet.getNumClasses();
//        LabelTranslator labelTranslator = dataSet.getLabelTranslator();
//        if (numClassesInTest>numClassesInTrain){
//            System.out.println("new labels not seen in the training set:");
//            for (int k=numClassesInTrain;k<numClassesInTest;k++){
//                System.out.println(""+k+"("+labelTranslator.toExtLabel(k)+")");
//            }
//        }
//        List<MultiLabel> predictions = boosting.predict(dataSet);
//        MultiLabel[] trueLabels = dataSet.getMultiLabels();
//
//        for (int k=0;k<numClassesInTest;k++){
//            PerClassMeasures perClassMeasures = new PerClassMeasures(trueLabels,predictions,k,labelTranslator.toExtLabel(k));
//            System.out.println("test: " + perClassMeasures);
//        }
//
//        int limit = config.getInt("test.analyzeMistakes.limit");
//        for (int i=0;i<dataSet.getNumDataPoints();i++){
//
//            MultiLabel prediction = predictions.get(i);
//            MultiLabel trueLabel = trueLabels[i];
//            if (!prediction.equals(trueLabel)){
//                System.out.println("=======================================");
//                Vector vector = dataSet.getRow(i);
//                System.out.println("data point "+i+" index id = "+dataSet.getDataPointSetting(i).getExtId());
//                if (trueLabel.outOfBound(numClassesInTrain)){
//                    System.out.println("true labels = "+trueLabel.toStringWithExtLabels(labelTranslator));
//                    System.out.println("predicted labels = "+prediction.toStringWithExtLabels(labelTranslator));
//                    System.out.println("it contains unseen labels");
//                } else{
////                    System.out.println(HMLGBInspector.analyzeMistake(boosting,vector,trueLabel,prediction,labelTranslator,limit));
//                }
//
//            }
//        }
//    }

    static int getNumClassesInTrain(Config config) throws Exception{
        MultiLabelClfDataSet dataSet = loadTrainData(config);
        int numClasses = dataSet.getNumClasses();
        return numClasses;
    }

}
