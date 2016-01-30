package edu.neu.ccs.pyramid.experiment;

import edu.neu.ccs.pyramid.configuration.Config;
import edu.neu.ccs.pyramid.dataset.*;
import edu.neu.ccs.pyramid.eval.Accuracy;
import edu.neu.ccs.pyramid.multilabel_classification.bmm_variant.BMMClassifier;
import edu.neu.ccs.pyramid.multilabel_classification.bmm_variant.BMMInitializer;
import edu.neu.ccs.pyramid.multilabel_classification.bmm_variant.BMMOptimizer;
import edu.neu.ccs.pyramid.util.MathUtil;
import edu.neu.ccs.pyramid.util.Sampling;

import java.io.File;
import java.io.IOException;
import java.nio.file.Paths;
import java.util.*;
import java.util.stream.Collectors;

/**
 * Created by Rainicy on 1/29/16.
 */
public class Exp226 {

    public static void main(String[] args) throws IOException, ClassNotFoundException {
        if (args.length != 1) {
            throw new IllegalArgumentException("Please specify a properties file.");
        }

        Config config = new Config(args[0]);

        System.out.println(config);

        if (config.getBoolean("splitData")) {
            splitData(config);
        }

        int repeat = config.getInt("repeat");
        int numFolder = config.getInt("numFolder");

        List<Integer> clusters = config.getIntegers("clusters");

        Map<Integer, Double> validationAccs = new HashMap<>();

        for (int cluster : clusters) {
            double avgAcc = 0.0;
            System.out.println("-------------- cluster " + cluster + "--------------");
            for (int f=0; f<numFolder; f++) {
                String folder = Paths.get(config.getString("output"), "folder"+f).toString();
                MultiLabelClfDataSet trainSet = TRECFormat.loadMultiLabelClfDataSet(new File(folder,"train"),
                        DataSetType.ML_CLF_SPARSE, true);
                MultiLabelClfDataSet testSet = TRECFormat.loadMultiLabelClfDataSet(new File(folder,"validation"),
                        DataSetType.ML_CLF_SPARSE, true);

                System.out.print("folder " + f + ": ");
                for (int r=0; r<repeat; r++) {
                    BMMClassifier bmmClassifier;
                    bmmClassifier = BMMClassifier.getBuilder()
                            .setNumClasses(trainSet.getNumClasses())
                            .setNumFeatures(trainSet.getNumFeatures())
                            .setNumClusters(cluster)
                            .setMultiClassClassifierType(config.getString("mixture.multiClassClassifierType"))
                            .setBinaryClassifierType(config.getString("mixture.binaryClassifierType"))
                            .build();
                    bmmClassifier.setPredictMode(config.getString("predict.mode"));
                    bmmClassifier.setAllowEmpty(config.getBoolean("predict.allowEmpty"));

                    BMMOptimizer optimizer = Exp211.getOptimizer(config, bmmClassifier, trainSet);
                    optimizer.setTemperature(1.0);

                    if (config.getBoolean("train.initialize")) {
                        BMMInitializer.initialize(bmmClassifier, trainSet, optimizer);
                    }

                    for (int i=0; i<config.getInt("em.numIterations"); i++) {
                        optimizer.iterate();
                    }
                    MultiLabel[] predict = bmmClassifier.predict(testSet);
                    double accuracy = Accuracy.accuracy(testSet.getMultiLabels(), predict);
                    avgAcc += accuracy;
                    System.out.print(accuracy + "\t");
                }
                System.out.println();
            }
            avgAcc = avgAcc / (double)numFolder / (double) repeat;
            validationAccs.put(cluster, avgAcc);
        }

        System.out.println("------------- tuning num clusters --------------");
        double maxAcc = Double.NEGATIVE_INFINITY;
        int bestCluster = -1;
        for (Map.Entry<Integer, Double> entry :  validationAccs.entrySet()) {
            if (entry.getValue() > maxAcc) {
                maxAcc = entry.getValue();
                bestCluster = entry.getKey();
            }
        }
        System.out.println("best cluster: " + bestCluster + "\tacc: " + validationAccs.get(bestCluster));
    }

    private static void splitData(Config config) throws IOException, ClassNotFoundException {
        MultiLabelClfDataSet dataSet = TRECFormat.loadMultiLabelClfDataSet(config.getString("input.data"),
                DataSetType.ML_CLF_SPARSE, true);

        double percent = config.getDouble("percentage");
        String output = config.getString("output");
        int numFolder = config.getInt("numFolder");

        for (int i=0; i<numFolder; i++) {
            String outputFolder = Paths.get(output, "folder"+i).toString();
            splitData(dataSet, percent, outputFolder);
        }
    }

    private static void splitData(MultiLabelClfDataSet dataSet, double percent, String outputFolder) {
        int[] arrayRange = MathUtil.range(0, dataSet.getNumDataPoints());
        List<Integer> range = Arrays.stream(arrayRange).boxed().collect(Collectors.toList());

        List<Integer> trainIndices = Sampling.sampleByPercentage(range, percent);
        Set<Integer> trainSet = new HashSet<>(trainIndices);
        List<Integer> validationIndices = new LinkedList<>();
        for (int i=0; i<dataSet.getNumDataPoints(); i++) {
            if (!trainSet.contains(i)) {
                validationIndices.add(i);
            }
        }

        MultiLabelClfDataSet trainData = DataSetUtil.sampleData(dataSet, trainIndices);
        MultiLabelClfDataSet testData = DataSetUtil.sampleData(dataSet, validationIndices);

        TRECFormat.save(trainData, new File(outputFolder, "train"));
        TRECFormat.save(testData, new File(outputFolder, "validation"));
    }
}
