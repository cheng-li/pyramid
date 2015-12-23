package edu.neu.ccs.pyramid.experiment;

import edu.neu.ccs.pyramid.configuration.Config;
import edu.neu.ccs.pyramid.dataset.DataSetType;
import edu.neu.ccs.pyramid.dataset.MultiLabel;
import edu.neu.ccs.pyramid.dataset.MultiLabelClfDataSet;
import edu.neu.ccs.pyramid.dataset.TRECFormat;
import edu.neu.ccs.pyramid.eval.Accuracy;
import edu.neu.ccs.pyramid.eval.Overlap;
import edu.neu.ccs.pyramid.multilabel_classification.bmm_variant.BMMClassifier;
import edu.neu.ccs.pyramid.multilabel_classification.bmm_variant.BMMInitializer;
import edu.neu.ccs.pyramid.multilabel_classification.bmm_variant.BMMOptimizer;

import java.io.File;
import java.util.List;

/**
 * Created by Rainicy on 12/9/15.
 */
public class Exp216 {
    public static void main(String[] args) throws Exception {
        if (args.length != 1) {
            throw new IllegalArgumentException("Please specify a properties file.");
        }

        Config config = new Config(args[0]);

        System.out.println(config);

        MultiLabelClfDataSet trainSet = TRECFormat.loadMultiLabelClfDataSet(config.getString("input.trainData"),
                DataSetType.ML_CLF_SPARSE, true);
        MultiLabelClfDataSet testSet = TRECFormat.loadMultiLabelClfDataSet(config.getString("input.testData"),
                DataSetType.ML_CLF_SPARSE, true);


        // how many interval iterations to predict.
        int interval = config.getInt("interval");
        int numIterations = config.getInt("numIterations");
        int numSamples = config.getInt("numSamples");
        String output = config.getString("output");
        List<Integer> clusters = config.getIntegers("numClusters");
        List<Double> softmaxVariances = config.getDoubles("softmaxVariances");
        List<Double> logitVariances = config.getDoubles("logitVariances");

        System.out.println(softmaxVariances);
        System.out.println(logitVariances);
        for (int numClusters : clusters) {
            for (double softmaxVariance : softmaxVariances) {
                for (double logitVariance : logitVariances) {
                    String modelName = "c" + numClusters + ".vs" + (int) softmaxVariance + ".vb" + (int) logitVariance + ".model";
                    System.out.println("----------------------------Model--------------------------");
                    System.out.println("-----------------<" + modelName + ">-----------------");

                    double v1 = Math.pow(10, softmaxVariance);
                    double v2 = Math.pow(10, logitVariance);
                    System.out.println("numClusters: " + numClusters);
                    System.out.println("softV: " + v1);
                    System.out.println("logitV: " + v2);

                    BMMClassifier bmmClassifier;
                    // loading from file
                    if (config.getBoolean("train.warmStart")) {
                        System.out.println("====================loading====================");
                        bmmClassifier = BMMClassifier.deserialize(new File(output, modelName));
                        bmmClassifier.setNumSample(numSamples);
                        bmmClassifier.setAllowEmpty(config.getBoolean("allowEmpty"));
                        bmmClassifier.setPredictMode(config.getString("predictMode"));
                    } else {
                        System.out.println("====================training====================");
                        bmmClassifier = new BMMClassifier(trainSet.getNumClasses(),numClusters,trainSet.getNumFeatures());
                        BMMOptimizer optimizer = new BMMOptimizer(bmmClassifier, trainSet,v1,v2);
                        bmmClassifier.setNumSample(numSamples);
                        bmmClassifier.setAllowEmpty(config.getBoolean("allowEmpty"));
                        bmmClassifier.setPredictMode(config.getString("predictMode"));

                        MultiLabel[] trainPredict;
                        MultiLabel[] testPredict;

                        if (config.getBoolean("initialize")) {
                            BMMInitializer.initialize(bmmClassifier,trainSet,v1,v2, new File(config.getString("initializeBy")));
                        }
                        else {
                            BMMInitializer.initialize(bmmClassifier,trainSet,v1,v2);
                        }
                        System.out.println("after initialization");
                        trainPredict = bmmClassifier.predict(trainSet);
                        testPredict = bmmClassifier.predict(testSet);

                        System.out.print("objective: "+optimizer.getObjective()+ "\t");
                        System.out.print("trainAcc : " + Accuracy.accuracy(trainSet.getMultiLabels(), trainPredict) + "\t");
                        System.out.print("trainOver: " + Overlap.overlap(trainSet.getMultiLabels(), trainPredict) + "\t");
                        System.out.print("testACC  : " + Accuracy.accuracy(testSet.getMultiLabels(), testPredict) + "\t");
                        System.out.println("testOver : "+ Overlap.overlap(testSet.getMultiLabels(), testPredict) + "\t");

                        for (int i=1;i<=numIterations;i++){
                            optimizer.iterate();
                            System.out.print("iter : "+i + "\t");
                            if ((i % interval) == 0) {
                                trainPredict = bmmClassifier.predict(trainSet);
                                testPredict = bmmClassifier.predict(testSet);
                                System.out.print("objective: "+optimizer.getTerminator().getLastValue() + "\t");
                                System.out.print("trainAcc : "+ Accuracy.accuracy(trainSet.getMultiLabels(),trainPredict)+ "\t");
                                System.out.print("trainOver: "+ Overlap.overlap(trainSet.getMultiLabels(), trainPredict)+ "\t");
                                System.out.print("testAcc  : "+ Accuracy.accuracy(testSet.getMultiLabels(),testPredict)+ "\t");
                                System.out.print("testOver : "+ Overlap.overlap(testSet.getMultiLabels(), testPredict)+ "\t");
                            }
                            System.out.println();
                        }
                    }

                    System.out.println("********************Results********************\n");
                    System.out.println();
                    System.out.print("trainAcc : " + Accuracy.accuracy(bmmClassifier, trainSet) + "\t");
                    System.out.print("trainOver: "+ Overlap.overlap(bmmClassifier, trainSet)+ "\t");
                    System.out.print("testAcc  : "+ Accuracy.accuracy(bmmClassifier,testSet)+ "\t");
                    System.out.println("testOver : "+ Overlap.overlap(bmmClassifier, testSet)+ "\t");
                    System.out.println();
                    System.out.println();
                    System.out.println(bmmClassifier);

                    if (config.getBoolean("saveModel")) {
                        (new File(output)).mkdirs();
                        File serializeModel = new File(output, modelName);
                        bmmClassifier.serialize(serializeModel);
                    }
                    System.out.println("\n\n\n\n");
                }
            }
        }

    }
}
