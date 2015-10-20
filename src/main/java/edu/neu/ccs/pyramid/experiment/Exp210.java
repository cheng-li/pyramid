package edu.neu.ccs.pyramid.experiment;

import edu.neu.ccs.pyramid.configuration.Config;
import edu.neu.ccs.pyramid.dataset.DataSetType;
import edu.neu.ccs.pyramid.dataset.MultiLabel;
import edu.neu.ccs.pyramid.dataset.MultiLabelClfDataSet;
import edu.neu.ccs.pyramid.dataset.TRECFormat;
import edu.neu.ccs.pyramid.eval.Accuracy;
import edu.neu.ccs.pyramid.eval.Overlap;
import edu.neu.ccs.pyramid.multilabel_classification.bmm.BMMClassifier;
import edu.neu.ccs.pyramid.multilabel_classification.bmm.BMMInitializer;
import edu.neu.ccs.pyramid.multilabel_classification.bmm.BMMOptimizer;

import java.io.File;
import java.util.Set;


/**
 * BMM multi-label 
 * Created by chengli on 10/8/15.
 */
public class Exp210 {
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

        int numClusters = config.getInt("numClusters");
        double variance = config.getDouble("variance");
        int numIterations = config.getInt("numIterations");
        int numSamples = config.getInt("numSamples");
        String output = config.getString("output");
        String modelName = config.getString("modelName");

        BMMClassifier bmmClassifier;
        if (config.getBoolean("train.warmStart")) {
            bmmClassifier = BMMClassifier.deserialize(new File(output, modelName));
        } else {
            bmmClassifier = new BMMClassifier(trainSet.getNumClasses(),numClusters,trainSet.getNumFeatures());
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
        }

        System.out.println("--------------------------------Results-----------------------------\n");
        System.out.println();
        System.out.print("trainAcc : " + Accuracy.accuracy(bmmClassifier, trainSet) + "\t");
        System.out.print("trainOver: "+ Overlap.overlap(bmmClassifier, trainSet)+ "\t");
        System.out.print("testAcc  : "+ Accuracy.accuracy(bmmClassifier,testSet)+ "\t");
        System.out.println("testOver : "+ Overlap.overlap(bmmClassifier, testSet)+ "\t");
        System.out.println();
        System.out.println();
        System.out.println(bmmClassifier);

        if (config.getBoolean("generateNewRate")) {
            Set<MultiLabel> samples = bmmClassifier.sampleFromSingles(config.getInt("topSample"));

            MultiLabel[] predictions = bmmClassifier.predict(trainSet);
            int cover = 0;
            for (MultiLabel l : predictions) {
                if (samples.contains(l)) {
                    cover += 1;
                }
            }
            System.out.println("Training cover rate: "  + (float)cover/ (float)predictions.length);

            predictions = bmmClassifier.predict(testSet);
            cover = 0;
            for (MultiLabel l : predictions) {
                if (samples.contains(l)) {
                    cover += 1;
                }
            }
            System.out.println("Testing cover rate: "  + (float)cover/ (float)predictions.length);
        }

        if (config.getBoolean("saveModel")) {
            File serializeModel = new File(output,modelName);
            bmmClassifier.serialize(serializeModel);
        }
    }
}
