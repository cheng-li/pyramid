package edu.neu.ccs.pyramid.simulation;


import edu.neu.ccs.pyramid.classification.logistic_regression.LogisticRegression;
import edu.neu.ccs.pyramid.dataset.MultiLabel;
import edu.neu.ccs.pyramid.dataset.MultiLabelClfDataSet;
import edu.neu.ccs.pyramid.eval.Accuracy;
import edu.neu.ccs.pyramid.eval.HammingLoss;
import edu.neu.ccs.pyramid.eval.Overlap;
import edu.neu.ccs.pyramid.multilabel_classification.bmm_variant.BMMClassifier;
import edu.neu.ccs.pyramid.multilabel_classification.bmm_variant.BMMInitializer;
import edu.neu.ccs.pyramid.multilabel_classification.bmm_variant.BMMOptimizer;

import java.util.Arrays;

public class MultiLabelSynthesizerTest {

    public static void main(String[] args) {
//        test1_br();
//        test1_mix();
//        test2_br();
        test2_mix();
    }

    private static void test1_br(){
        MultiLabelClfDataSet trainSet = MultiLabelSynthesizer.randomBinary();
        MultiLabelClfDataSet testSet = MultiLabelSynthesizer.randomBinary();
        int numClusters = 1;
        double softmaxVariance = 100;
        double logitVariance = 100;
        BMMClassifier bmmClassifier = new BMMClassifier(trainSet.getNumClasses(),numClusters,trainSet.getNumFeatures());
        BMMOptimizer optimizer = new BMMOptimizer(bmmClassifier, trainSet,softmaxVariance,logitVariance);
        bmmClassifier.setAllowEmpty(true);
        bmmClassifier.setPredictMode("dynamic");
        BMMInitializer.initialize(bmmClassifier, trainSet, softmaxVariance, logitVariance);
        for (int i=1;i<=10;i++){
            optimizer.iterate();
            MultiLabel[] trainPredict;
            MultiLabel[] testPredict;
            trainPredict = bmmClassifier.predict(trainSet);
            testPredict = bmmClassifier.predict(testSet);
            System.out.print("iter : "+i + "\t");
            System.out.print("objective: "+optimizer.getTerminator().getLastValue() + "\t");
            System.out.println("train Hamming loss : "+ HammingLoss.hammingLoss(bmmClassifier,trainSet)+ "\t");
            System.out.print("trainAcc : "+ Accuracy.accuracy(trainSet.getMultiLabels(), trainPredict)+ "\t");
            System.out.print("trainOver: "+ Overlap.overlap(trainSet.getMultiLabels(), trainPredict)+ "\t");
            System.out.println("test Hamming loss : "+ HammingLoss.hammingLoss(bmmClassifier,testSet)+ "\t");
            System.out.print("testAcc  : "+ Accuracy.accuracy(testSet.getMultiLabels(),testPredict)+ "\t");
            System.out.println("testOver : "+ Overlap.overlap(testSet.getMultiLabels(), testPredict)+ "\t");

        }
    }

    private static void test1_mix(){
        MultiLabelClfDataSet trainSet = MultiLabelSynthesizer.randomBinary();
        MultiLabelClfDataSet testSet = MultiLabelSynthesizer.randomBinary();
        int numClusters = 2;
        double softmaxVariance = 100;
        double logitVariance = 100;
        BMMClassifier bmmClassifier = new BMMClassifier(trainSet.getNumClasses(),numClusters,trainSet.getNumFeatures());
        BMMOptimizer optimizer = new BMMOptimizer(bmmClassifier, trainSet,softmaxVariance,logitVariance);
        bmmClassifier.setAllowEmpty(true);
        bmmClassifier.setPredictMode("dynamic");
        BMMInitializer.initialize(bmmClassifier, trainSet, softmaxVariance, logitVariance);
        for (int i=1;i<=10;i++){
            optimizer.iterate();
            MultiLabel[] trainPredict;
            MultiLabel[] testPredict;
            trainPredict = bmmClassifier.predict(trainSet);
            testPredict = bmmClassifier.predict(testSet);
            System.out.print("iter : "+i + "\t");
            System.out.print("objective: "+optimizer.getTerminator().getLastValue() + "\t");
            System.out.println("train Hamming loss : " + HammingLoss.hammingLoss(bmmClassifier, trainSet) + "\t");
            System.out.print("trainAcc : "+ Accuracy.accuracy(trainSet.getMultiLabels(), trainPredict)+ "\t");
            System.out.print("trainOver: "+ Overlap.overlap(trainSet.getMultiLabels(), trainPredict)+ "\t");
            System.out.println("test Hamming loss : " + HammingLoss.hammingLoss(bmmClassifier, testSet) + "\t");
            System.out.print("testAcc  : " + Accuracy.accuracy(testSet.getMultiLabels(), testPredict) + "\t");
            System.out.println("testOver : "+ Overlap.overlap(testSet.getMultiLabels(), testPredict)+ "\t");
        }
    }

    private static void test2_br(){
        MultiLabelClfDataSet trainSet = MultiLabelSynthesizer.randomTwoLabels();
        MultiLabelClfDataSet testSet = MultiLabelSynthesizer.randomTwoLabels();
        int numClusters = 1;
        double softmaxVariance = 100;
        double logitVariance = 100;
        BMMClassifier bmmClassifier = new BMMClassifier(trainSet.getNumClasses(),numClusters,trainSet.getNumFeatures());
        BMMOptimizer optimizer = new BMMOptimizer(bmmClassifier, trainSet,softmaxVariance,logitVariance);
        bmmClassifier.setAllowEmpty(true);
        bmmClassifier.setPredictMode("dynamic");
        BMMInitializer.initialize(bmmClassifier, trainSet, softmaxVariance, logitVariance);
        for (int i=1;i<=1;i++){
            optimizer.iterate();
            MultiLabel[] trainPredict;
            MultiLabel[] testPredict;
            trainPredict = bmmClassifier.predict(trainSet);
            testPredict = bmmClassifier.predict(testSet);
            System.out.print("iter : "+i + "\t");
            System.out.print("objective: "+optimizer.getTerminator().getLastValue() + "\t");
            System.out.println("train Hamming loss : " + HammingLoss.hammingLoss(bmmClassifier, trainSet) + "\t");
            System.out.print("trainAcc : "+ Accuracy.accuracy(trainSet.getMultiLabels(), trainPredict)+ "\t");
            System.out.print("trainOver: "+ Overlap.overlap(trainSet.getMultiLabels(), trainPredict)+ "\t");
            System.out.println("test Hamming loss : " + HammingLoss.hammingLoss(bmmClassifier, testSet) + "\t");
            System.out.print("testAcc  : " + Accuracy.accuracy(testSet.getMultiLabels(), testPredict) + "\t");
            System.out.println("testOver : "+ Overlap.overlap(testSet.getMultiLabels(), testPredict)+ "\t");
        }
        System.out.println(bmmClassifier.predict(trainSet.getRow(99)));
        System.out.println((LogisticRegression)bmmClassifier.getBinaryClassifiers()[0][0]);
        System.out.println((LogisticRegression)bmmClassifier.getBinaryClassifiers()[0][1]);
        System.out.println(Arrays.toString(trainSet.getMultiLabels()));
    }

    private static void test2_mix(){
        MultiLabelClfDataSet trainSet = MultiLabelSynthesizer.randomTwoLabels();
        MultiLabelClfDataSet testSet = MultiLabelSynthesizer.randomTwoLabels();
        int numClusters = 3;
        double softmaxVariance = 100;
        double logitVariance = 100;
        BMMClassifier bmmClassifier = new BMMClassifier(trainSet.getNumClasses(),numClusters,trainSet.getNumFeatures());
        BMMOptimizer optimizer = new BMMOptimizer(bmmClassifier, trainSet,softmaxVariance,logitVariance);
        bmmClassifier.setAllowEmpty(true);
        bmmClassifier.setPredictMode("dynamic");
        BMMInitializer.initialize(bmmClassifier, trainSet, softmaxVariance, logitVariance);
        for (int i=1;i<=10;i++){
            optimizer.iterate();
            MultiLabel[] trainPredict;
            MultiLabel[] testPredict;
            trainPredict = bmmClassifier.predict(trainSet);
            testPredict = bmmClassifier.predict(testSet);
            System.out.print("iter : "+i + "\t");
            System.out.print("objective: "+optimizer.getTerminator().getLastValue() + "\t");
            System.out.println("train Hamming loss : " + HammingLoss.hammingLoss(bmmClassifier, trainSet) + "\t");
            System.out.print("trainAcc : "+ Accuracy.accuracy(trainSet.getMultiLabels(), trainPredict)+ "\t");
            System.out.print("trainOver: "+ Overlap.overlap(trainSet.getMultiLabels(), trainPredict)+ "\t");
            System.out.println("test Hamming loss : " + HammingLoss.hammingLoss(bmmClassifier, testSet) + "\t");
            System.out.print("testAcc  : " + Accuracy.accuracy(testSet.getMultiLabels(), testPredict) + "\t");
            System.out.println("testOver : "+ Overlap.overlap(testSet.getMultiLabels(), testPredict)+ "\t");

        }
        System.out.println(bmmClassifier.predict(trainSet.getRow(99)));
        System.out.println((LogisticRegression)bmmClassifier.getBinaryClassifiers()[0][0]);
        System.out.println((LogisticRegression)bmmClassifier.getBinaryClassifiers()[0][1]);
        System.out.println(Arrays.toString(trainSet.getMultiLabels()));
    }

}