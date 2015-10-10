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
        int numSamples = config.getInt("numSamples");

        BMMClassifier bmmClassifier = new BMMClassifier(trainSet.getNumClasses(),numClusters,trainSet.getNumFeatures());
        BMMOptimizer optimizer = new BMMOptimizer(bmmClassifier,trainSet,variance);

        bmmClassifier.setNumSample(numSamples);
        System.out.print("random init" + "\t" );
        System.out.print("trainAcc : "+ Accuracy.accuracy(bmmClassifier, trainSet) + "\t");
        System.out.print("trainOver: "+ Overlap.overlap(bmmClassifier, trainSet) + "\t");
        System.out.print("testACC  : "+ Accuracy.accuracy(bmmClassifier,testSet) + "\t");
        System.out.println("testOver : "+ Overlap.overlap(bmmClassifier, testSet) + "\t");

        if (config.getBoolean("initialize")){
            BMMInitializer bmmInitializer = new BMMInitializer();
            bmmInitializer.initialize(bmmClassifier,trainSet);
            System.out.print("pure-label" + "\t");
            System.out.print("trainAcc : "+ Accuracy.accuracy(bmmClassifier,trainSet)+ "\t");
            System.out.print("trainOver: "+ Overlap.overlap(bmmClassifier, trainSet)+ "\t");
            System.out.print("testAcc  : "+ Accuracy.accuracy(bmmClassifier,testSet)+ "\t");
            System.out.println("testOver : "+ Overlap.overlap(bmmClassifier, testSet)+ "\t");
        }



        for (int i=1;i<=numIterations;i++){
            optimizer.iterate();
            System.out.print("iter : "+i + "\t");
            System.out.print("objective: "+optimizer.getTerminator().getLastValue() + "\t");
            System.out.print("trainAcc : "+ Accuracy.accuracy(bmmClassifier,trainSet)+ "\t");
            System.out.print("trainOver: "+ Overlap.overlap(bmmClassifier, trainSet)+ "\t");
            System.out.print("testAcc  : "+ Accuracy.accuracy(bmmClassifier,testSet)+ "\t");
            System.out.println("testOver : "+ Overlap.overlap(bmmClassifier, testSet)+ "\t");
        }

        System.out.println("history = "+optimizer.getTerminator().getHistory());
        System.out.println(bmmClassifier);
    }
}
