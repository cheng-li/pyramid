//package edu.neu.ccs.pyramid.experiment;
//
//import edu.neu.ccs.pyramid.configuration.Config;
//import edu.neu.ccs.pyramid.dataset.DataSetType;
//import edu.neu.ccs.pyramid.dataset.MultiLabel;
//import edu.neu.ccs.pyramid.dataset.MultiLabelClfDataSet;
//import edu.neu.ccs.pyramid.dataset.TRECFormat;
//import edu.neu.ccs.pyramid.eval.*;
//import edu.neu.ccs.pyramid.multilabel_classification.imlgb.IMLGradientBoosting;
//import edu.neu.ccs.pyramid.multilabel_classification.sampling.GibbsSampling;
//import edu.neu.ccs.pyramid.multilabel_classification.sampling.GibbsSamplingConfig;
//import edu.neu.ccs.pyramid.multilabel_classification.sampling.GibbsSamplingTrainer;
//import edu.neu.ccs.pyramid.regression.regression_tree.LeafOutputType;
//import org.elasticsearch.common.StopWatch;
//
//import java.io.File;
//import java.io.IOException;
//import java.util.Arrays;
//import java.util.OptionalDouble;
//import java.util.stream.DoubleStream;
//import java.util.stream.IntStream;
//
///**
// * Created by Rainicy on 9/7/15.
// */
//public class Exp207 {
//    public static void main(String[] args) throws Exception {
//
//        Config config = new Config(args[0]);
//        System.out.println(config);
//
//        // loading data
//        MultiLabelClfDataSet trainSet = loadData(config, config.getString("input.trainData"));
//
//        // build the config
//        GibbsSamplingConfig trainConfig = new GibbsSamplingConfig.Builder(trainSet)
//                .numLeaves(config.getInt("numLeaves"))
//                .learningRate(config.getDouble("learningRate"))
//                .numSplitIntervals(config.getInt("numSplitIntervals"))
//                .minDataPerLeaf(config.getInt("minDataPerLeaf"))
//                .dataSamplingRate(config.getInt("dataSamplingRate"))
//                .featureSamplingRate(config.getInt("featureSamplingRate"))
//                .randomLevel(config.getInt("randomLevel"))
//                .considerHardTree(config.getBoolean("considerHardTree"))
//                .considerExpectationTree(config.getBoolean("considerExpectationTree"))
//                .considerProbabilisticTree(config.getBoolean("considerProbabilisticTree"))
//                .setLeafOutputType(LeafOutputType.AVERAGE)
//                .numRounds(config.getInt("numRounds"))
//                .build();
//
//        StopWatch stopWatch;
//        // loading an old sampling model or training a new model
//        GibbsSampling sampling;
//        if (config.getBoolean("train.warmStart")) {
//            System.out.println("Loading model...");
//            sampling = GibbsSampling.deserialize(new File(config.getString("output.folder"),
//                    config.getString("model.name")));
//        } else {
//            System.out.println("Training model...");
//            sampling = new GibbsSampling(trainSet.getNumClasses(),
//                    config.getInt("K"), config.getInt("lastK"));
//            GibbsSamplingTrainer trainer = new GibbsSamplingTrainer(trainConfig, sampling);
//
//            stopWatch = new StopWatch();
//            stopWatch.start();
//            trainer.train();
//            stopWatch.stop();
//            System.out.println("Training time: " + stopWatch);
//        }
//
//
//        // before prediction, setup the gibbs sampling parameters
//        sampling.setK(config.getInt("K"));
//        sampling.setLastK(config.getInt("lastK"));
//
//        System.out.println("Training is done...");
//
//        // if report the results on train or test
//        if (config.getBoolean("train")) {
//            int numClasses = trainSet.getNumClasses();
//            MultiLabel[] multiLabels = trainSet.getMultiLabels();
//
//            MultiLabel[] predictions;
//            stopWatch = new StopWatch();
//            stopWatch.start();
//            if (config.getBoolean("sampling.warmStart")) {
//                IMLGradientBoosting boosting = IMLGradientBoosting.deserialize(new File(
//                        config.getString("warmStart.model")));
//                MultiLabel[] predMultiLabel = boosting.predict(trainSet);
//
//                if (config.getBoolean("tuning.sampling")) {
//                    predictions = sampling.tuningPredict(trainSet, predMultiLabel);
//                } else {
//                    predictions = sampling.predict(trainSet, predMultiLabel);
//                }
//
//            } else {
//                predictions = sampling.predict(trainSet);
//            }
//            stopWatch.stop();
//            System.out.println("========Train.result======");
//            System.out.println("Testing time: " + stopWatch);
//
//
//            MicroMeasures microMeasures = new MicroMeasures(numClasses);
//            MacroMeasures macroMeasures = new MacroMeasures(numClasses);
//            microMeasures.update(multiLabels,predictions);
//            macroMeasures.update(multiLabels,predictions);
//            System.out.println("data-hamming loss:\t" + HammingLoss.hammingLoss(multiLabels, predictions, numClasses));
//            System.out.println("data-accuracy:\t" + Accuracy.accuracy(multiLabels, predictions));
//            System.out.println("data-overlap\t"+ Overlap.overlap(multiLabels,predictions));
//            System.out.println("data-precision:\t" + Precision.precision(multiLabels, predictions));
//            System.out.println("data-recall:\t" + Recall.recall(multiLabels,predictions));
//            System.out.println("label-macro-measures = \n" + macroMeasures);
//            System.out.println("label-micro-measures = \n" + microMeasures);
//        }
//        if (config.getBoolean("test")) {
//            MultiLabelClfDataSet testSet = loadData(config, config.getString("input.testData"));
//            int numClasses = testSet.getNumClasses();
//            MultiLabel[] multiLabels = testSet.getMultiLabels();
//
//            stopWatch = new StopWatch();
//            stopWatch.start();
//            MultiLabel[] predictions;
//            if (config.getBoolean("sampling.warmStart")) {
//                IMLGradientBoosting boosting = IMLGradientBoosting.deserialize(new File(
//                        config.getString("warmStart.model")));
//                MultiLabel[] predMultiLabel = boosting.predict(testSet);
//
//                if (config.getBoolean("tuning.sampling")) {
//                    System.out.println("prediction by tuning prediction...");
//                    predictions = sampling.tuningPredict(testSet, predMultiLabel);
//                } else {
//                    predictions = sampling.predict(testSet, predMultiLabel);
//                }
//            } else {
//                predictions = sampling.predict(testSet);
//            }
//            stopWatch.stop();
//            System.out.println("========Test.result======");
//            System.out.println("Testing time: " + stopWatch);
//
////            System.out.println("TRUE: " + Arrays.toString(multiLabels));
////            System.out.println("PRED: " + Arrays.toString(predictions));
//
//
//            MicroMeasures microMeasures = new MicroMeasures(numClasses);
//            MacroMeasures macroMeasures = new MacroMeasures(numClasses);
//            microMeasures.update(multiLabels,predictions);
//            macroMeasures.update(multiLabels,predictions);
//            System.out.println("data-hamming loss:\t" + HammingLoss.hammingLoss(multiLabels, predictions, numClasses));
//            System.out.println("data-accuracy:\t" + Accuracy.accuracy(multiLabels, predictions));
//            System.out.println("data-overlap:\t"+ Overlap.overlap(multiLabels,predictions));
//            System.out.println("data-precision:\t" + Precision.precision(multiLabels, predictions));
//            System.out.println("data-recall:\t " + Recall.recall(multiLabels,predictions));
//            System.out.println("label-macro-measures = \n" + macroMeasures);
//            System.out.println("label-micro-measures = \n" + microMeasures);
//        }
//
//
//        // serialize model
//        File serializedModel =  new File(config.getString("output.folder"),config.getString("model.name"));
//        sampling.serialize(serializedModel);
//    }
//
//    static MultiLabelClfDataSet loadData(Config config, String dataName) throws IOException, ClassNotFoundException {
//        File dataFile = new File(new File(config.getString("input.folder"),"data_sets"),dataName);
//        MultiLabelClfDataSet dataSet = TRECFormat.loadMultiLabelClfDataSet(dataFile, DataSetType.ML_CLF_SPARSE,
//                true);
//        return dataSet;
//    }
//}