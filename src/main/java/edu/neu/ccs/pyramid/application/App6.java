package edu.neu.ccs.pyramid.application;

import edu.neu.ccs.pyramid.configuration.Config;
import edu.neu.ccs.pyramid.dataset.DataSetType;
import edu.neu.ccs.pyramid.dataset.MultiLabel;
import edu.neu.ccs.pyramid.dataset.MultiLabelClfDataSet;
import edu.neu.ccs.pyramid.dataset.TRECFormat;
import edu.neu.ccs.pyramid.eval.Accuracy;
import edu.neu.ccs.pyramid.eval.FMeasure;
import edu.neu.ccs.pyramid.eval.MLMeasures;
import edu.neu.ccs.pyramid.eval.Overlap;
import edu.neu.ccs.pyramid.multilabel_classification.crf.CMLCRF;
import edu.neu.ccs.pyramid.multilabel_classification.crf.CRFLoss;
import edu.neu.ccs.pyramid.multilabel_classification.crf.PluginF1;
import edu.neu.ccs.pyramid.optimization.GradientDescent;
import edu.neu.ccs.pyramid.optimization.LBFGS;

import java.io.File;
import java.util.concurrent.TimeUnit;

/**
 * pair-wise CRF
 * Created by chengli on 8/19/16.
 */
public class App6 {
    public static void main(String[] args) throws Exception {
        if (args.length != 1) {
            throw new IllegalArgumentException("Please specify a properties file.");
        }

        Config config = new Config(args[0]);

        System.out.println(config);

        MultiLabelClfDataSet trainSet = TRECFormat.loadMultiLabelClfDataSet(config.getString("input.trainData"),
                DataSetType.ML_CLF_SEQ_SPARSE, true);
        MultiLabelClfDataSet testSet = TRECFormat.loadMultiLabelClfDataSet(config.getString("input.testData"),
                DataSetType.ML_CLF_SEQ_SPARSE, true);
        double gaussianVariance = config.getDouble("gaussianVariance");

        // loading or save model infos.
        String output = config.getString("output");
        String modelName = config.getString("modelName");

        CMLCRF cmlcrf = null;

        if (config.getString("train.warmStart").equals("true")) {
            cmlcrf = CMLCRF.deserialize(new File(output, modelName));
            System.out.println("loading model:");
            System.out.println(cmlcrf);
        } else if (config.getString("train.warmStart").equals("auto")) {
            cmlcrf = CMLCRF.deserialize(new File(output, modelName));
            System.out.println("retrain model:");
            CRFLoss crfLoss = new CRFLoss(cmlcrf, trainSet, gaussianVariance);
            crfLoss.setParallelism(true);
            crfLoss.setRegularizeAll(config.getBoolean("regularizeAll"));
            train(crfLoss, cmlcrf, trainSet, testSet, config);

        } else if (config.getString("train.warmStart").equals("false")) {
            cmlcrf = new CMLCRF(trainSet);
            cmlcrf.setConsiderPair(config.getBoolean("considerLabelPair"));
            CRFLoss crfLoss = new CRFLoss(cmlcrf, trainSet, gaussianVariance);
            crfLoss.setParallelism(true);
            crfLoss.setRegularizeAll(config.getBoolean("regularizeAll"));
            train(crfLoss, cmlcrf, trainSet, testSet, config);
        }

        System.out.println();
        System.out.println();
        System.out.println("--------------------------------Results-----------------------------\n");
        MLMeasures measures = new MLMeasures(cmlcrf, trainSet);
        System.out.println("========== Train ==========\n");
        System.out.println(measures);

        System.out.println("========== Test ==========\n");
        long startTimePred = System.nanoTime();
        MultiLabel[] preds = cmlcrf.predict(testSet);
        long stopTimePred = System.nanoTime();
        long predTime = stopTimePred - startTimePred;
        System.out.println("\nprediction time: " + TimeUnit.NANOSECONDS.toSeconds(predTime) + " sec.");
        System.out.println(new MLMeasures(cmlcrf, testSet));
        System.out.println("\n\n");
        PluginF1 pluginF1 = new PluginF1(cmlcrf);
        System.out.println("Plugin F1");
        System.out.println(new MLMeasures(pluginF1, testSet));

        if (config.getBoolean("saveModel")) {
            (new File(output)).mkdirs();
            File serializeModel = new File(output, modelName);
            cmlcrf.serialize(serializeModel);
        }
    }

    private static void train(CRFLoss crfLoss, CMLCRF cmlcrf, MultiLabelClfDataSet trainSet, MultiLabelClfDataSet testSet, Config config) {
        MultiLabel[] predTrain;
        MultiLabel[] predTest;
        if (config.getBoolean("isLBFGS")) {
            long startTime = System.nanoTime();
            LBFGS optimizer = new LBFGS(crfLoss);
            optimizer.getTerminator().setAbsoluteEpsilon(0.1);

            for (int i=0; i<config.getInt("numRounds"); i++) {
                optimizer.iterate();
                predTrain = cmlcrf.predict(trainSet);
                predTest = cmlcrf.predict(testSet);
                System.out.print("iter: "+ String.format("%04d", i));
                System.out.print("\tobjective: "+ String.format("%.4f", optimizer.getTerminator().getLastValue()));
                System.out.print("\tTrain acc: " + String.format("%.4f", Accuracy.accuracy(trainSet.getMultiLabels(), predTrain)));
                System.out.print("\tTrain overlap " + String.format("%.4f", Overlap.overlap(trainSet.getMultiLabels(), predTrain)));
                System.out.print("\tTrain F1 " + String.format("%.4f",Overlap.overlap(trainSet.getMultiLabels(), predTrain)));
                System.out.print("\tTest acc: " + String.format("%.4f", FMeasure.f1(testSet.getMultiLabels(), predTest)));
                System.out.print("\tTest overlap " + String.format("%.4f",Overlap.overlap(testSet.getMultiLabels(), predTest)));
                System.out.println("\tTest F1 " + String.format("%.4f",FMeasure.f1(testSet.getMultiLabels(), predTest)));
            }

            long stopTime = System.nanoTime();
            long trainTime = stopTime - startTime;
            System.out.println("\ntraining time: " + TimeUnit.NANOSECONDS.toSeconds(trainTime) + " sec.");

        } else {
            GradientDescent optimizer = new GradientDescent(crfLoss);
            for (int i=0; i<config.getInt("numRounds"); i++) {
                optimizer.iterate();
                predTrain = cmlcrf.predict(trainSet);
                predTest = cmlcrf.predict(testSet);
                System.out.print("iter: "+ String.format("%04d", i));
                System.out.print("\tTrain acc: " + String.format("%.4f",Accuracy.accuracy(trainSet.getMultiLabels(), predTrain)));
                System.out.print("\tTrain overlap " + String.format("%.4f",Overlap.overlap(trainSet.getMultiLabels(), predTrain)));
                System.out.print("\tTest acc: " + String.format("%.4f",Accuracy.accuracy(testSet.getMultiLabels(), predTest)));
                System.out.println("\tTest overlap " + String.format("%.4f",Overlap.overlap(testSet.getMultiLabels(), predTest)));
            }
        }
    }
}
