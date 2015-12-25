package edu.neu.ccs.pyramid.experiment;

import edu.neu.ccs.pyramid.configuration.Config;
import edu.neu.ccs.pyramid.dataset.DataSetType;
import edu.neu.ccs.pyramid.dataset.MultiLabel;
import edu.neu.ccs.pyramid.dataset.MultiLabelClfDataSet;
import edu.neu.ccs.pyramid.dataset.TRECFormat;
import edu.neu.ccs.pyramid.eval.Accuracy;
import edu.neu.ccs.pyramid.eval.Overlap;
import edu.neu.ccs.pyramid.multilabel_classification.crf.CMLCRF;
import edu.neu.ccs.pyramid.multilabel_classification.crf.CRFLoss;
import edu.neu.ccs.pyramid.optimization.GradientDescent;
import edu.neu.ccs.pyramid.optimization.LBFGS;
import edu.neu.ccs.pyramid.optimization.Optimizer;

import java.io.File;
import java.io.IOException;

/**
 * Created by Rainicy on 12/22/15.
 */
public class Exp218 {
    public static void main(String[] args) throws Exception {
        if (args.length != 1) {
            throw new IllegalArgumentException("Please specify a properties file.");
        }

        Config config = new Config(args[0]);

        System.out.println(config);

        MultiLabelClfDataSet trainSet = TRECFormat.loadMultiLabelClfDataSet(config.getString("input.trainData"),
                DataSetType.ML_CLF_DENSE, true);
        MultiLabelClfDataSet testSet = TRECFormat.loadMultiLabelClfDataSet(config.getString("input.testData"),
                DataSetType.ML_CLF_DENSE, true);
        double gaussianVariance = config.getDouble("gaussianVariance");

        // loading or save model infos.
        String output = config.getString("output");
        String modelName = config.getString("modelName");

        CMLCRF cmlcrf;
        MultiLabel[] predTrain;
        MultiLabel[] predTest;

        if (config.getBoolean("train.warmStart")) {
            cmlcrf = CMLCRF.deserialize(new File(output, modelName));
            System.out.println("loading model:");
            System.out.println(cmlcrf);
        } else{
            cmlcrf = new CMLCRF(trainSet);
            CRFLoss crfLoss = new CRFLoss(cmlcrf, trainSet, gaussianVariance);

            if (config.getBoolean("isLBFGS")) {
                LBFGS optimizer = new LBFGS(crfLoss);
                optimizer.getTerminator().setAbsoluteEpsilon(0.1);

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

        System.out.println();
        System.out.println();
        System.out.println("--------------------------------Results-----------------------------\n");
        predTrain = cmlcrf.predict(trainSet);
        predTest = cmlcrf.predict(testSet);
        System.out.print("Train acc: " + String.format("%.4f",Accuracy.accuracy(trainSet.getMultiLabels(), predTrain)));
        System.out.print("\tTrain overlap " + String.format("%.4f",Overlap.overlap(trainSet.getMultiLabels(), predTrain)));
        System.out.print("\tTest acc: " + String.format("%.4f",Accuracy.accuracy(testSet.getMultiLabels(), predTest)));
        System.out.println("\tTest overlap " + String.format("%.4f",Overlap.overlap(testSet.getMultiLabels(), predTest)));

        if (config.getBoolean("saveModel")) {
            (new File(output)).mkdirs();
            File serializeModel = new File(output, modelName);
            cmlcrf.serialize(serializeModel);
        }
    }
}
