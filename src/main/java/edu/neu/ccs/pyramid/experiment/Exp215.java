package edu.neu.ccs.pyramid.experiment;

import edu.neu.ccs.pyramid.configuration.Config;
import edu.neu.ccs.pyramid.dataset.DataSetType;
import edu.neu.ccs.pyramid.dataset.MultiLabel;
import edu.neu.ccs.pyramid.dataset.MultiLabelClfDataSet;
import edu.neu.ccs.pyramid.dataset.TRECFormat;
import edu.neu.ccs.pyramid.eval.Accuracy;
import edu.neu.ccs.pyramid.eval.Overlap;
import edu.neu.ccs.pyramid.multilabel_classification.powerset.LPClassifier;
import edu.neu.ccs.pyramid.multilabel_classification.powerset.LPOptimizer;

import java.io.File;

/**
 * Created by Rainicy on 12/3/15.
 */
public class Exp215 {

    public static void main(String[] args) throws Exception {
        if (args.length != 1) {
            throw new IllegalArgumentException("Please specify a properties file.");
        }

        Config config = new Config(args[0]);

        System.out.println(config);

        MultiLabelClfDataSet trainSet;
        MultiLabelClfDataSet testSet;
        String matrixType = config.getString("input.matrixType");

        switch (matrixType){
            case "sparse_random":
                trainSet= TRECFormat.loadMultiLabelClfDataSet(config.getString("input.trainData"),
                        DataSetType.ML_CLF_SPARSE, true);
                testSet = TRECFormat.loadMultiLabelClfDataSet(config.getString("input.testData"),
                        DataSetType.ML_CLF_SPARSE, true);
                break;
            case "sparse_sequential":
                trainSet= TRECFormat.loadMultiLabelClfDataSet(config.getString("input.trainData"),
                        DataSetType.ML_CLF_SEQ_SPARSE, true);
                testSet = TRECFormat.loadMultiLabelClfDataSet(config.getString("input.testData"),
                        DataSetType.ML_CLF_SEQ_SPARSE, true);
                break;
            case "dense":
                trainSet= TRECFormat.loadMultiLabelClfDataSet(config.getString("input.trainData"),
                        DataSetType.ML_CLF_DENSE, true);
                testSet = TRECFormat.loadMultiLabelClfDataSet(config.getString("input.testData"),
                        DataSetType.ML_CLF_DENSE, true);
                break;
            default:
                throw new IllegalArgumentException("unknown type");
        }


        LPClassifier lpClassifier;

        String output = config.getString("output");
        String modelName = config.getString("modelName");
        if (config.getBoolean("train.warmStart")) {
            lpClassifier = LPClassifier.deserialize(new File(output, modelName));
        } else {
            lpClassifier = new LPClassifier(trainSet);
            LPOptimizer optimizer = new LPOptimizer(lpClassifier,trainSet);
            optimizer.optimize(config);
        }

//        System.out.println("classifier: \n" + lpClassifier);

        MultiLabel[] trainPredict = lpClassifier.predict(trainSet);
        MultiLabel[] testPredict = lpClassifier.predict(testSet);
        System.out.println("--------------------------------Results-----------------------------\n");
        System.out.println();
        System.out.print("trainAcc : " + Accuracy.accuracy(trainSet.getMultiLabels(), trainPredict) + "\t");
        System.out.print("trainOver: "+ Overlap.overlap(trainSet.getMultiLabels(), trainPredict)+ "\t");
        System.out.print("testAcc  : "+ Accuracy.accuracy(testSet.getMultiLabels(),testPredict)+ "\t");
        System.out.println("testOver : "+ Overlap.overlap(testSet.getMultiLabels(), testPredict)+ "\t");
        System.out.println();
        System.out.println();

        if (config.getBoolean("saveModel")) {
            (new File(output)).mkdirs();
            File serializeModel = new File(output, modelName);
            lpClassifier.serialize(serializeModel);
        }
    }
}
