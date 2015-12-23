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

import java.io.IOException;

/**
 * Created by Rainicy on 12/22/15.
 */
public class Exp218 {
    public static void main(String[] args) throws IOException, ClassNotFoundException {
        if (args.length != 1) {
            throw new IllegalArgumentException("Please specify a properties file.");
        }

        Config config = new Config(args[0]);

        System.out.println(config);

        MultiLabelClfDataSet trainSet = TRECFormat.loadMultiLabelClfDataSet(config.getString("input.trainData"),
                DataSetType.ML_CLF_SPARSE, true);
        MultiLabelClfDataSet testSet = TRECFormat.loadMultiLabelClfDataSet(config.getString("input.testData"),
                DataSetType.ML_CLF_SPARSE, true);
        double gaussianVariance = config.getDouble("gaussianVariance");

        CMLCRF cmlcrf = new CMLCRF(trainSet);
        CRFLoss crfLoss = new CRFLoss(cmlcrf, trainSet, gaussianVariance);

        MultiLabel[] predTrain;
        MultiLabel[] predTest;
        if (config.getBoolean("isLBFGS")) {
            LBFGS optimizer = new LBFGS(crfLoss);
            optimizer.getTerminator().setAbsoluteEpsilon(0.1);

            for (int i=0; i<config.getInt("numRounds"); i++) {
                optimizer.iterate();
                predTrain = cmlcrf.predict(trainSet);
                predTest = cmlcrf.predict(testSet);
                System.out.print("iter: "+ i);
                System.out.print("\tTrain acc: " + Accuracy.accuracy(trainSet.getMultiLabels(), predTrain));
                System.out.print("\tTrain overlap " + Overlap.overlap(trainSet.getMultiLabels(), predTrain));
                System.out.print("\tTest acc: " + Accuracy.accuracy(testSet.getMultiLabels(), predTest));
                System.out.println("\tTest overlap " + Overlap.overlap(testSet.getMultiLabels(), predTest));
            }


        } else {
            GradientDescent optimizer = new GradientDescent(crfLoss);
            for (int i=0; i<config.getInt("numRounds"); i++) {
                optimizer.iterate();
                predTrain = cmlcrf.predict(trainSet);
                predTest = cmlcrf.predict(testSet);
                System.out.print("iter: "+ i);
                System.out.print("\tTrain acc: " + Accuracy.accuracy(trainSet.getMultiLabels(), predTrain));
                System.out.print("\tTrain overlap " + Overlap.overlap(trainSet.getMultiLabels(), predTrain));
                System.out.print("\tTest acc: " + Accuracy.accuracy(testSet.getMultiLabels(), predTest));
                System.out.println("\tTest overlap " + Overlap.overlap(testSet.getMultiLabels(), predTest));
            }
        }

    }
}
