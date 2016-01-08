package edu.neu.ccs.pyramid.experiment;

import edu.neu.ccs.pyramid.classification.Classifier;
import edu.neu.ccs.pyramid.classification.logistic_regression.LogisticRegression;
import edu.neu.ccs.pyramid.configuration.Config;
import edu.neu.ccs.pyramid.dataset.DataSetType;
import edu.neu.ccs.pyramid.dataset.MultiLabel;
import edu.neu.ccs.pyramid.dataset.MultiLabelClfDataSet;
import edu.neu.ccs.pyramid.dataset.TRECFormat;
import edu.neu.ccs.pyramid.eval.Accuracy;
import edu.neu.ccs.pyramid.eval.HammingLoss;
import edu.neu.ccs.pyramid.eval.Overlap;
import edu.neu.ccs.pyramid.multilabel_classification.bmm_variant.BMMClassifier;
import edu.neu.ccs.pyramid.multilabel_classification.bmm_variant.BMMOptimizer;
import edu.neu.ccs.pyramid.util.Serialization;
import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.Vector;

/**
 * average learned mixture classifiers
 * Created by chengli on 12/25/15.
 */
public class Exp139 {
    public static void main(String[] args) throws Exception{
        if (args.length != 1) {
            throw new IllegalArgumentException("Please specify a properties file.");
        }

        Config config = new Config(args[0]);

        System.out.println(config);
        MultiLabelClfDataSet trainSet = TRECFormat.loadMultiLabelClfDataSet(config.getString("input.trainData"), DataSetType.ML_CLF_SPARSE, true);
        MultiLabelClfDataSet testSet = TRECFormat.loadMultiLabelClfDataSet(config.getString("input.testData"), DataSetType.ML_CLF_SPARSE, true);

        BMMClassifier bmmClassifier = loadModel(config);
        MultiLabel[] trainPredict;
        MultiLabel[] testPredict;
        trainPredict = bmmClassifier.predict(trainSet);
        testPredict = bmmClassifier.predict(testSet);
        System.out.print("train Hamming loss : " + HammingLoss.hammingLoss(bmmClassifier, trainSet) + "\t");
        System.out.print("trainAcc : "+ Accuracy.accuracy(trainSet.getMultiLabels(), trainPredict)+ "\t");
        System.out.print("trainOver: "+ Overlap.overlap(trainSet.getMultiLabels(), trainPredict)+ "\t");
        System.out.print("test Hamming loss : " + HammingLoss.hammingLoss(bmmClassifier, testSet) + "\t");
        System.out.print("testAcc  : " + Accuracy.accuracy(testSet.getMultiLabels(), testPredict) + "\t");
        System.out.println("testOver : "+ Overlap.overlap(testSet.getMultiLabels(), testPredict)+ "\t");

        average(bmmClassifier);
        trainPredict = bmmClassifier.predict(trainSet);
        testPredict = bmmClassifier.predict(testSet);
        System.out.println("after averaging");
        System.out.print("train Hamming loss : " + HammingLoss.hammingLoss(bmmClassifier, trainSet) + "\t");
        System.out.print("trainAcc : "+ Accuracy.accuracy(trainSet.getMultiLabels(), trainPredict)+ "\t");
        System.out.print("trainOver: "+ Overlap.overlap(trainSet.getMultiLabels(), trainPredict)+ "\t");
        System.out.print("test Hamming loss : " + HammingLoss.hammingLoss(bmmClassifier, testSet) + "\t");
        System.out.print("testAcc  : " + Accuracy.accuracy(testSet.getMultiLabels(), testPredict) + "\t");
        System.out.println("testOver : "+ Overlap.overlap(testSet.getMultiLabels(), testPredict)+ "\t");

        System.out.println("optimization");

        double softmaxVariance = config.getDouble("softmaxVariance");
        double logitVariance = config.getDouble("logitVariance");
        int numIterations = config.getInt("numIterations");

        BMMOptimizer optimizer = new BMMOptimizer(bmmClassifier, trainSet);
        optimizer.setPriorVarianceBinary(logitVariance);
        optimizer.setPriorVarianceMultiClass(softmaxVariance);

        for (int i=1;i<=numIterations;i++){
            optimizer.iterate();
            trainPredict = bmmClassifier.predict(trainSet);
            testPredict = bmmClassifier.predict(testSet);
            System.out.print("iter : "+i + "\t");
            System.out.print("objective: "+optimizer.getTerminator().getLastValue() + "\t");
            System.out.print("trainAcc : "+ Accuracy.accuracy(trainSet.getMultiLabels(),trainPredict)+ "\t");
            System.out.print("trainOver: "+ Overlap.overlap(trainSet.getMultiLabels(), trainPredict)+ "\t");
            System.out.print("testAcc  : "+ Accuracy.accuracy(testSet.getMultiLabels(),testPredict)+ "\t");
            System.out.println("testOver : "+ Overlap.overlap(testSet.getMultiLabels(), testPredict)+ "\t");
        }

    }

    private static void average(BMMClassifier bmmClassifier){
        for (int l=0;l<bmmClassifier.getNumClasses();l++){
            average(bmmClassifier,l);
        }
    }

    private static void average(BMMClassifier bmm, int label){
        Classifier.ProbabilityEstimator[][] logistics = bmm.getBinaryClassifiers();
        int numClusters = bmm.getNumClusters();
        int numFeatures =  ((LogisticRegression)logistics[0][0]).getNumFeatures();
        Vector negativeAverageVector = new DenseVector(numFeatures);
        Vector positiveAverageVector = new DenseVector(numFeatures);
        for (int k=0;k<numClusters;k++){
            Vector negativeVector = ((LogisticRegression) logistics[k][label]).getWeights().getWeightsWithoutBiasForClass(0);
            negativeAverageVector = negativeAverageVector.plus(negativeVector);
            Vector positiveVector = ((LogisticRegression) logistics[k][label]).getWeights().getWeightsWithoutBiasForClass(1);
            positiveAverageVector = positiveAverageVector.plus(positiveVector);
        }

        negativeAverageVector = negativeAverageVector.divide(numClusters);
        positiveAverageVector = positiveAverageVector.divide(numClusters);

        for (int k=0;k<numClusters;k++){
            Vector negativeVector = ((LogisticRegression) logistics[k][label]).getWeights().getWeightsWithoutBiasForClass(0);
            negativeVector.assign(negativeAverageVector);
            Vector positiveVector = ((LogisticRegression) logistics[k][label]).getWeights().getWeightsWithoutBiasForClass(1);
            positiveVector.assign(positiveAverageVector);
        }
    }

    private static BMMClassifier loadModel(Config config) throws Exception{
        String model = config.getString("model");
        BMMClassifier bmm = (BMMClassifier) Serialization.deserialize(model);
        return bmm;
    }
}
