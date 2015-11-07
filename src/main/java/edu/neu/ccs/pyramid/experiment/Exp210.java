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
import java.util.HashSet;
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
        BMMOptimizer optimizer;
        if (config.getBoolean("train.warmStart")) {
            bmmClassifier = BMMClassifier.deserialize(new File(output, modelName));
            optimizer = BMMOptimizer.deserialize(new File(output, modelName+".optimizer"));
        } else {
            bmmClassifier = new BMMClassifier(trainSet.getNumClasses(),numClusters,trainSet.getNumFeatures());
            optimizer = new BMMOptimizer(bmmClassifier,trainSet,variance);
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
            System.out.println("total samples: " + samples.size());

            MultiLabel[] predictions = bmmClassifier.predict(trainSet);
            int cover = 0;
            Set<MultiLabel> uniqueTrainPred = new HashSet<>();
            for (MultiLabel l : predictions) {
                if (samples.contains(l)) {
                    cover += 1;
                }
                if (!uniqueTrainPred.contains(l)) {
                    uniqueTrainPred.add(l);
                }
            }
            Set<MultiLabel> uniqueTrainY = new HashSet<>();
            for (MultiLabel l : trainSet.getMultiLabels()) {
                if (!uniqueTrainY.contains(l)) {
                    uniqueTrainY.add(l);
                }
            }
            System.out.println("Training unique prediction combinations: " + uniqueTrainPred.size());
            System.out.println("Training unique label combinations: " + uniqueTrainY.size());
            System.out.println("Training cover rate: "  + (float)cover/ (float)predictions.length);

            predictions = bmmClassifier.predict(testSet);
            cover = 0;
            Set<MultiLabel> uniqueTestPred = new HashSet<>();
            for (MultiLabel l : predictions) {
                if (samples.contains(l)) {
                    cover += 1;
                }
                if (!uniqueTestPred.contains(l)) {
                    uniqueTestPred.add(l);
                }
            }
            Set<MultiLabel> uniqueTestY = new HashSet<>();
            for (MultiLabel l : testSet.getMultiLabels()) {
                if (!uniqueTestY.contains(l)) {
                    uniqueTestY.add(l);
                }
            }

            System.out.println("Testing unique prediction combinations: " + uniqueTestPred.size());
            System.out.println("Testing unique label combinations: " + uniqueTestY.size());
            System.out.println("Testing cover rate: "  + (float)cover/ (float)predictions.length);
        }

        if (config.getBoolean("saveModel")) {
            File serializeModel = new File(output,modelName);
            bmmClassifier.serialize(serializeModel);
            optimizer.serialize(new File(output, modelName+".optimizer"));
        }
    }
}
