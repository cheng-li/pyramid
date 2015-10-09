package edu.neu.ccs.pyramid.experiment;

import edu.neu.ccs.pyramid.configuration.Config;
import edu.neu.ccs.pyramid.dataset.DataSetType;
import edu.neu.ccs.pyramid.dataset.MultiLabelClfDataSet;
import edu.neu.ccs.pyramid.dataset.TRECFormat;
import edu.neu.ccs.pyramid.eval.Accuracy;
import edu.neu.ccs.pyramid.eval.Overlap;
import edu.neu.ccs.pyramid.multilabel_classification.bmm.BMMClassifier;
import edu.neu.ccs.pyramid.multilabel_classification.bmm.BMMInitializer;
import edu.neu.ccs.pyramid.multilabel_classification.bmm.BMMOptimizer;

import java.io.IOException;

/**
 * BMM multi-label 
 * Created by chengli on 10/8/15.
 */
public class Exp210 {
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

        int numClusters = config.getInt("numClusters");
        double variance = config.getDouble("variance");
        int numIterations = config.getInt("numIterations");

        BMMClassifier bmmClassifier = new BMMClassifier(trainSet.getNumClasses(),numClusters,trainSet.getNumFeatures());
        BMMOptimizer optimizer = new BMMOptimizer(bmmClassifier,trainSet,variance);

        System.out.println("after random initialization");
        System.out.println("train acc = "+ Accuracy.accuracy(bmmClassifier, trainSet));
        System.out.println("train overlap = "+ Overlap.overlap(bmmClassifier, trainSet));
        System.out.println("test acc = "+ Accuracy.accuracy(bmmClassifier,testSet));
        System.out.println("test overlap = "+ Overlap.overlap(bmmClassifier, testSet));

        if (config.getBoolean("initialize")){
            BMMInitializer bmmInitializer = new BMMInitializer();
            bmmInitializer.initialize(bmmClassifier,trainSet);
            System.out.println("after pure-label clustering initialization");
            System.out.println("train acc = "+ Accuracy.accuracy(bmmClassifier, trainSet));
            System.out.println("train overlap = "+ Overlap.overlap(bmmClassifier, trainSet));
            System.out.println("test acc = "+ Accuracy.accuracy(bmmClassifier,testSet));
            System.out.println("test overlap = "+ Overlap.overlap(bmmClassifier, testSet));
        }



        for (int i=1;i<=numIterations;i++){
            optimizer.iterate();
            System.out.println("after iteration "+i);
            System.out.println("objective = "+optimizer.getTerminator().getLastValue());
            System.out.println("train acc = "+ Accuracy.accuracy(bmmClassifier,trainSet));
            System.out.println("train overlap = "+ Overlap.overlap(bmmClassifier, trainSet));
            System.out.println("test acc = "+ Accuracy.accuracy(bmmClassifier,testSet));
            System.out.println("test overlap = "+ Overlap.overlap(bmmClassifier, testSet));
        }

        System.out.println("history = "+optimizer.getTerminator().getHistory());
        System.out.println(bmmClassifier);
    }
}
