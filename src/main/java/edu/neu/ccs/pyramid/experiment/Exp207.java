package edu.neu.ccs.pyramid.experiment;

import edu.neu.ccs.pyramid.configuration.Config;
import edu.neu.ccs.pyramid.dataset.DataSetType;
import edu.neu.ccs.pyramid.dataset.MultiLabel;
import edu.neu.ccs.pyramid.dataset.MultiLabelClfDataSet;
import edu.neu.ccs.pyramid.dataset.TRECFormat;
import edu.neu.ccs.pyramid.eval.*;
import edu.neu.ccs.pyramid.multilabel_classification.sampling.GibbsSampling;
import edu.neu.ccs.pyramid.multilabel_classification.sampling.GibbsSamplingConfig;
import edu.neu.ccs.pyramid.multilabel_classification.sampling.GibbsSamplingTrainer;
import edu.neu.ccs.pyramid.regression.regression_tree.LeafOutputType;
import org.elasticsearch.common.StopWatch;
import org.elasticsearch.common.base.Stopwatch;

import java.io.File;
import java.io.IOException;

/**
 * Created by Rainicy on 9/7/15.
 */
public class Exp207 {
    public static void main(String[] args) throws IOException, ClassNotFoundException {
        if (args.length != 1) {
            throw new IllegalArgumentException("Please specify a properties file.");
        }

        Config config = new Config(args[0]);
        System.out.println(config);

        MultiLabelClfDataSet trainSet = loadData(config, config.getString("input.trainData"));

        GibbsSampling sampling = new GibbsSampling(trainSet.getNumClasses(), config.getInt("gibbsSamplingTimes"));

        GibbsSamplingConfig trainConfig = new GibbsSamplingConfig.Builder(trainSet)
                .numLeaves(config.getInt("numLeaves"))
                .learningRate(config.getDouble("learningRate"))
                .numSplitIntervals(config.getInt("numSplitIntervals"))
                .minDataPerLeaf(config.getInt("minDataPerLeaf"))
                .dataSamplingRate(config.getInt("dataSamplingRate"))
                .featureSamplingRate(config.getInt("featureSamplingRate"))
                .randomLevel(config.getInt("randomLevel"))
                .considerHardTree(config.getBoolean("considerHardTree"))
                .considerExpectationTree(config.getBoolean("considerExpectationTree"))
                .considerProbabilisticTree(config.getBoolean("considerProbabilisticTree"))
                .setLeafOutputType(LeafOutputType.AVERAGE)
                .numRounds(config.getInt("numRounds"))
                .build();

        GibbsSamplingTrainer trainer = new GibbsSamplingTrainer(trainConfig, sampling);

        StopWatch stopWatch = new StopWatch();
        stopWatch.start();
        trainer.train();
        stopWatch.stop();
        System.out.println("Training time: " + stopWatch);

        MultiLabelClfDataSet testSet = loadData(config, config.getString("input.testData"));
        int numClasses = testSet.getNumClasses();
        MultiLabel[] multiLabels = testSet.getMultiLabels();

        stopWatch = new StopWatch();
        stopWatch.start();
        MultiLabel[] predictions = sampling.predict(testSet);
        stopWatch.stop();
        System.out.println("Testing time: " + stopWatch);

        MicroMeasures microMeasures = new MicroMeasures(numClasses);
        MacroMeasures macroMeasures = new MacroMeasures(numClasses);
        microMeasures.update(multiLabels,predictions);
        macroMeasures.update(multiLabels,predictions);
        System.out.println("data-hamming loss = " + HammingLoss.hammingLoss(multiLabels, predictions, numClasses));
        System.out.println("data-accuracy = " + Accuracy.accuracy(multiLabels, predictions));
//        System.out.println("proportion accuracy on data set = " + Accuracy.partialAccuracy(multiLabels, predictions)); // same as overlap
        System.out.println("data-precision = " + Precision.precision(multiLabels, predictions));
        System.out.println("data-recall = " + Recall.recall(multiLabels,predictions));
        System.out.println("data-overlap = "+ Overlap.overlap(multiLabels,predictions));
//        System.out.println("data-average precision= " + AveragePrecision.averagePrecisionByProbs(sampling,testSet));
        System.out.println("label-macro-measures = \n" + macroMeasures);
        System.out.println("label-micro-measures = \n" + microMeasures);
    }

    static MultiLabelClfDataSet loadData(Config config, String dataName) throws IOException, ClassNotFoundException {
        File dataFile = new File(new File(config.getString("input.folder"),
                "data_sets"),dataName);
        MultiLabelClfDataSet dataSet = TRECFormat.loadMultiLabelClfDataSet(dataFile, DataSetType.ML_CLF_SPARSE,
                true);
        return dataSet;
    }
}
