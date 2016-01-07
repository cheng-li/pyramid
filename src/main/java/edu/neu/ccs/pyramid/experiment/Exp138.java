package edu.neu.ccs.pyramid.experiment;

import edu.neu.ccs.pyramid.configuration.Config;
import edu.neu.ccs.pyramid.dataset.DataSetType;
import edu.neu.ccs.pyramid.dataset.MultiLabel;
import edu.neu.ccs.pyramid.dataset.MultiLabelClfDataSet;
import edu.neu.ccs.pyramid.dataset.TRECFormat;
import edu.neu.ccs.pyramid.eval.Accuracy;
import edu.neu.ccs.pyramid.eval.HammingLoss;
import edu.neu.ccs.pyramid.eval.Overlap;
import edu.neu.ccs.pyramid.multilabel_classification.bmm_variant.BMMClassifier;
import edu.neu.ccs.pyramid.multilabel_classification.bmm_variant.BMMInitializer;
import edu.neu.ccs.pyramid.multilabel_classification.bmm_variant.BMMOptimizer;
import edu.neu.ccs.pyramid.util.Grid;

import java.io.File;
import java.util.List;

/**
 * deterministic annealing
 * Created by chengli on 12/18/15.
 */
public class Exp138 {
    public static void main(String[] args) throws Exception{
        if (args.length != 1) {
            throw new IllegalArgumentException("Please specify a properties file.");
        }

        Config config = new Config(args[0]);

        System.out.println(config);

        System.out.println("deterministic annealing EM");
        MultiLabelClfDataSet trainSet = TRECFormat.loadMultiLabelClfDataSet(config.getString("input.trainData"), DataSetType.ML_CLF_SPARSE, true);
        MultiLabelClfDataSet testSet = TRECFormat.loadMultiLabelClfDataSet(config.getString("input.testData"), DataSetType.ML_CLF_SPARSE, true);
        int numClusters = config.getInt("numClusters");
        double softmaxVariance = config.getDouble("clusterVariance");
        double logitVariance = config.getDouble("predictionVariance");
        double start = config.getDouble("startInverseT");
        int numTemperatures = config.getInt("numTemperatures");

        List<Double> grid = Grid.uniform(start, 1, numTemperatures);
        BMMClassifier bmmClassifier = BMMClassifier.getBuilder()
                .setNumClasses(trainSet.getNumClasses())
                .setNumFeatures(trainSet.getNumFeatures())
                .setNumClusters(numClusters)
                .setBinaryClassifierType("lr")
                .setMultiClassClassifierType("lr")
                .build();

        bmmClassifier.setAllowEmpty(config.getBoolean("allowEmpty"));
        bmmClassifier.setPredictMode("dynamic");
        BMMInitializer.initialize(bmmClassifier, trainSet, softmaxVariance, logitVariance);

        int numIterations = config.getInt("numIterations");

        for (double it: grid){
            BMMOptimizer optimizer = new BMMOptimizer(bmmClassifier, trainSet,softmaxVariance,logitVariance);
            optimizer.setInverseTemperature(it);
            System.out.println("inverse temperature = "+it);
            for (int i=1;i<=numIterations;i++){
                optimizer.iterate();
                MultiLabel[] trainPredict;
                MultiLabel[] testPredict;
                trainPredict = bmmClassifier.predict(trainSet);
                testPredict = bmmClassifier.predict(testSet);
                System.out.print("iter : "+i + "\t");
                System.out.print("objective: "+optimizer.getTerminator().getLastValue() + "\t");
                System.out.print("train Hamming loss : " + HammingLoss.hammingLoss(bmmClassifier, trainSet) + "\t");
                System.out.print("trainAcc : "+ Accuracy.accuracy(trainSet.getMultiLabels(), trainPredict)+ "\t");
                System.out.print("trainOver: "+ Overlap.overlap(trainSet.getMultiLabels(), trainPredict)+ "\t");
                System.out.print("test Hamming loss : " + HammingLoss.hammingLoss(bmmClassifier, testSet) + "\t");
                System.out.print("testAcc  : " + Accuracy.accuracy(testSet.getMultiLabels(), testPredict) + "\t");
                System.out.println("testOver : "+ Overlap.overlap(testSet.getMultiLabels(), testPredict)+ "\t");

            }
        }
    }
}
