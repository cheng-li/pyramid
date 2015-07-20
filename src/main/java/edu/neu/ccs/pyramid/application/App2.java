package edu.neu.ccs.pyramid.application;

import com.fasterxml.jackson.databind.ObjectMapper;
import edu.neu.ccs.pyramid.configuration.Config;
import edu.neu.ccs.pyramid.dataset.*;
import edu.neu.ccs.pyramid.eval.Accuracy;
import edu.neu.ccs.pyramid.eval.Overlap;
import edu.neu.ccs.pyramid.feature.TopFeatures;
import edu.neu.ccs.pyramid.feature_selection.FeatureDistribution;
import edu.neu.ccs.pyramid.multilabel_classification.MultiLabelPredictionAnalysis;
import edu.neu.ccs.pyramid.multilabel_classification.imlgb.IMLGBConfig;
import edu.neu.ccs.pyramid.multilabel_classification.imlgb.IMLGBInspector;
import edu.neu.ccs.pyramid.multilabel_classification.imlgb.IMLGBTrainer;
import edu.neu.ccs.pyramid.multilabel_classification.imlgb.IMLGradientBoosting;
import edu.neu.ccs.pyramid.util.Serialization;
import org.apache.commons.io.FileUtils;
import org.apache.commons.lang3.time.StopWatch;

import java.io.File;
import java.util.ArrayList;
import java.util.Collection;
import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

/**
 * imlgb
 * follow exp14
 * Created by chengli on 6/13/15.
 */
public class App2 {

    public static void main(String[] args) throws Exception{
        if (args.length !=1){
            throw new IllegalArgumentException("please specify the config file");
        }

        Config config = new Config(args[0]);
        System.out.println(config);

        new File(config.getString("output.folder")).mkdirs();

        if (config.getBoolean("train")){
            train(config);
            report(config,config.getString("input.trainData"));
        }

        if (config.getBoolean("test")){
            report(config,config.getString("input.testData"));
        }
    }

    public static void main(Config config) throws Exception{
        new File(config.getString("output.folder")).mkdirs();

        if (config.getBoolean("train")){
            train(config);
            report(config,config.getString("input.trainData"));
        }

        if (config.getBoolean("test")){
            report(config,config.getString("input.testData"));
        }
    }

    static MultiLabelClfDataSet loadData(Config config, String dataName) throws Exception{
        File dataFile = new File(new File(config.getString("input.folder"),
                "data_sets"),dataName);
        MultiLabelClfDataSet dataSet = TRECFormat.loadMultiLabelClfDataSet(dataFile, DataSetType.ML_CLF_SPARSE,
                true);
        return dataSet;
    }

//    static MultiLabelClfDataSet loadTrainData(Config config) throws Exception{
//        String trainFile = new File(config.getString("input.folder"),
//                config.getString("input.trainData")).getAbsolutePath();
//        MultiLabelClfDataSet dataSet;
//
//        if (config.getBoolean("input.featureMatrix.sparse")){
//            dataSet= TRECFormat.loadMultiLabelClfDataSet(new File(trainFile), DataSetType.ML_CLF_SPARSE,
//                    true);
//        } else {
//            dataSet= TRECFormat.loadMultiLabelClfDataSet(new File(trainFile), DataSetType.ML_CLF_DENSE,
//                    true);
//        }
//
//        return dataSet;
//    }
//
//    static MultiLabelClfDataSet loadTestData(Config config) throws Exception{
//        String trainFile = new File(config.getString("input.folder"),
//                config.getString("input.testData")).getAbsolutePath();
//        MultiLabelClfDataSet dataSet;
//
//        if (config.getBoolean("input.featureMatrix.sparse")){
//            dataSet= TRECFormat.loadMultiLabelClfDataSet(new File(trainFile), DataSetType.ML_CLF_SPARSE,
//                    true);
//        } else {
//            dataSet= TRECFormat.loadMultiLabelClfDataSet(new File(trainFile), DataSetType.ML_CLF_DENSE,
//                    true);
//        }
//
//        return dataSet;
//    }

    static void train(Config config) throws Exception{
        String output = config.getString("output.folder");
        int numIterations = config.getInt("train.numIterations");
        int numLeaves = config.getInt("train.numLeaves");
        double learningRate = config.getDouble("train.learningRate");
        int minDataPerLeaf = config.getInt("train.minDataPerLeaf");
        String modelName = "model";
        double featureSamplingRate = config.getDouble("train.featureSamplingRate");
        double dataSamplingRate = config.getDouble("train.dataSamplingRate");

        StopWatch stopWatch = new StopWatch();
        stopWatch.start();

        MultiLabelClfDataSet dataSet = loadData(config,config.getString("input.trainData"));

        MultiLabelClfDataSet testSet = null;
        if (config.getBoolean("train.showTrainProgress")){
            testSet = loadData(config,config.getString("input.testData"));
        }


        int[] activeFeatures = IntStream.range(0, dataSet.getNumFeatures()).toArray();

        int numClasses = dataSet.getNumClasses();
        System.out.println("number of class = "+numClasses);
        IMLGBConfig imlgbConfig = new IMLGBConfig.Builder(dataSet)
                .dataSamplingRate(dataSamplingRate)
                .featureSamplingRate(featureSamplingRate)
                .learningRate(learningRate)
                .minDataPerLeaf(minDataPerLeaf)
                .numLeaves(numLeaves)
                .numSplitIntervals(config.getInt("train.numSplitIntervals"))
                .usePrior(config.getBoolean("train.usePrior"))
                .build();

        IMLGradientBoosting boosting;
        if (config.getBoolean("train.warmStart")){
            boosting = IMLGradientBoosting.deserialize(new File(output,modelName));
        } else {
            boosting  = new IMLGradientBoosting(numClasses);
        }

        String predictFashion = config.getString("predict.fashion").toLowerCase();
        switch (predictFashion){
            case "crf":
                boosting.setPredictFashion(IMLGradientBoosting.PredictFashion.CRF);
                break;
            case "independent":
                boosting.setPredictFashion(IMLGradientBoosting.PredictFashion.INDEPENDENT);
                break;
            default:
                throw new IllegalArgumentException("predict.fashion should be independent or crf");
        }

        IMLGBTrainer trainer = new IMLGBTrainer(imlgbConfig,boosting);

        //todo make it better
        trainer.setActiveFeatures(activeFeatures);

        for (int i=0;i<numIterations;i++){
            System.out.println("iteration "+i);
            trainer.iterate();
            System.out.println("model size = "+boosting.getRegressors(0).size());
            if (config.getBoolean("train.showTrainProgress")){
                System.out.println("accuracy on training set = "+ Accuracy.accuracy(boosting,
                        dataSet));
                System.out.println("overlap on training set = "+ Overlap.overlap(boosting, dataSet));
            }
            if (config.getBoolean("train.showTestProgress")){
                System.out.println("accuracy on test set = "+ Accuracy.accuracy(boosting,
                        testSet));
                System.out.println("overlap on test set = "+ Overlap.overlap(boosting, testSet));
            }
        }
        File serializedModel =  new File(output,modelName);


        boosting.serialize(serializedModel);
        System.out.println(stopWatch);

    }

    static void report(Config config, String dataName) throws Exception{
        System.out.println("generating reports for data set "+dataName);
        String output = config.getString("output.folder");
        String modelName = "model";
        File analysisFolder = new File(new File(output,"reports"),dataName+"_reports");
        analysisFolder.mkdirs();
        FileUtils.cleanDirectory(analysisFolder);

        IMLGradientBoosting boosting = IMLGradientBoosting.deserialize(new File(output,modelName));
        String predictFashion = config.getString("predict.fashion").toLowerCase();
        switch (predictFashion){
            case "crf":
                boosting.setPredictFashion(IMLGradientBoosting.PredictFashion.CRF);
                break;
            case "independent":
                boosting.setPredictFashion(IMLGradientBoosting.PredictFashion.INDEPENDENT);
                break;
        }

        MultiLabelClfDataSet dataSet = loadData(config,dataName);

        System.out.println("accuracy on data set = "+Accuracy.accuracy(boosting,dataSet));
        System.out.println("overlap on data set = "+ Overlap.overlap(boosting,dataSet));
//        System.out.println("macro-averaged measure on training set:");
//        System.out.println(new MacroAveragedMeasures(boosting,dataSet));
        if (true){
            File distributionFile = new File(new File(config.getString("input.folder"), "meta_data"),"distributions.ser");
            Collection<FeatureDistribution> distributions = (Collection) Serialization.deserialize(distributionFile);
            int limit = config.getInt("report.topFeatures.limit");
            List<TopFeatures> topFeaturesList = IntStream.range(0,boosting.getNumClasses())
                    .mapToObj(k -> IMLGBInspector.topFeatures(boosting, k, limit, distributions))
                    .collect(Collectors.toList());
            ObjectMapper mapper = new ObjectMapper();
            String file = "top_features.json";
            mapper.writeValue(new File(analysisFolder,file), topFeaturesList);

        }

//        if (config.getBoolean("verify.topNgramsFeatures")){
//            for (int k=0;k<dataSet.getNumClasses();k++) {
//                List<Feature> featureNames = IMLGBInspector.topFeatures(boosting, k)
//                        .stream().filter(feature -> feature instanceof Ngram)
//                        .map(feature -> (Ngram)feature)
//                        .filter(ngram -> ngram.getN()>1)
//                        .collect(Collectors.toList());
//                System.out.println("top ngram features for class " + k + "(" + labelTranslator.toExtLabel(k) + "):");
//                System.out.println(featureNames);
//            }
//        }

        if (true){
            int limit = config.getInt("report.rule.limit");
            int numDocsPerFile = config.getInt("report.numDocsPerFile");
            int numFiles = (int)Math.ceil((double)dataSet.getNumDataPoints()/numDocsPerFile);

            double probThreshold=config.getDouble("report.classProbThreshold");

            for (int i=0;i<numFiles;i++){
                int start = i*numDocsPerFile;
                int end = start+numDocsPerFile;
                List<MultiLabelPredictionAnalysis> partition = new ArrayList<>();
                for (int a=start;a<end && a<dataSet.getNumDataPoints();a++){
                    List<Integer> classes = new ArrayList<Integer>();
                    for (int k = 0; k < boosting.getNumClasses(); k++){
                        if (boosting.predictClassProb(dataSet.getRow(a),k)>=probThreshold){
                            classes.add(k);
                        }
                    }
                    partition.add(IMLGBInspector.analyzePrediction(boosting, dataSet, a, classes, limit));
                }
                ObjectMapper mapper = new ObjectMapper();

                String file = "report_"+(i+1)+".json";
                mapper.writeValue(new File(analysisFolder,file), partition);
            }

        }
        System.out.println("reports generated");
    }




}
