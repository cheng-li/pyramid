package edu.neu.ccs.pyramid.simulation;


import edu.neu.ccs.pyramid.classification.logistic_regression.LogisticRegression;
import edu.neu.ccs.pyramid.configuration.Config;
import edu.neu.ccs.pyramid.dataset.*;
import edu.neu.ccs.pyramid.eval.Accuracy;
import edu.neu.ccs.pyramid.eval.HammingLoss;
import edu.neu.ccs.pyramid.eval.Overlap;
import edu.neu.ccs.pyramid.multilabel_classification.bmm_variant.*;
import edu.neu.ccs.pyramid.multilabel_classification.multi_label_logistic_regression.MLLogisticRegression;
import edu.neu.ccs.pyramid.multilabel_classification.multi_label_logistic_regression.MLLogisticTrainer;
import edu.neu.ccs.pyramid.multilabel_classification.powerset.LPClassifier;
import edu.neu.ccs.pyramid.multilabel_classification.powerset.LPOptimizer;

import java.io.File;
import java.util.Arrays;
import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

public class MultiLabelSynthesizerTest {
    private static final Config config = new Config("config/local.config");
    private static final String DATASETS = config.getString("input.datasets");
    private static final String TMP = config.getString("output.tmp");

    public static void main(String[] args) throws Exception{
//        test1_br();
//        test1_mix();
//        test2_br();
//        test2_mix();
//        test3_br();
//        test3_mix();
//        test4Dump();
//        test4_br();
//        test4_crf();
//        test4_powerset();
//        test4_mix();
//        test5Dump();
//                test5_br();
//        test5_powerset();
//        test5_mix();
//                test6Dump();
//                test6_br();
//        test6_powerset();
        test6_mix();
//        test6_mix_boost();
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

    private static void test3_br(){
        MultiLabelClfDataSet trainSet = MultiLabelSynthesizer.randomMultiClass();
        MultiLabelClfDataSet testSet = MultiLabelSynthesizer.randomMultiClass();
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

    private static void test3_mix(){
        MultiLabelClfDataSet trainSet = MultiLabelSynthesizer.randomMultiClass();
        MultiLabelClfDataSet testSet = MultiLabelSynthesizer.randomMultiClass();
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


    private static void test4Dump(){
        MultiLabelClfDataSet all = MultiLabelSynthesizer.flipOne(10000,2,3);
        List<Integer> trainIndices = IntStream.range(0,5000).mapToObj(i -> i).collect(Collectors.toList());
        List<Integer> testIndices = IntStream.range(5000,10000).mapToObj(i->i).collect(Collectors.toList());
        DataSetUtil.sampleData(all,trainIndices);
        MultiLabelClfDataSet trainSet = DataSetUtil.sampleData(all, trainIndices);
        MultiLabelClfDataSet testSet =  DataSetUtil.sampleData(all, testIndices);
        TRECFormat.save(trainSet, new File(TMP,"train.trec"));
        TRECFormat.save(testSet, new File(TMP,"test.trec"));
    }


    private static void test4_br() throws Exception{
        System.out.println("binary");
        MultiLabelClfDataSet trainSet = TRECFormat.loadMultiLabelClfDataSet(new File(DATASETS,"simulation/multi-label/flip_one/2_labels/train.trec"), DataSetType.ML_CLF_DENSE,true);
        MultiLabelClfDataSet testSet = TRECFormat.loadMultiLabelClfDataSet(new File(DATASETS, "simulation/multi-label/flip_one/2_labels/test.trec"), DataSetType.ML_CLF_DENSE, true);
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
            System.out.print("train Hamming loss : " + HammingLoss.hammingLoss(bmmClassifier, trainSet) + "\t");
            System.out.print("trainAcc : "+ Accuracy.accuracy(trainSet.getMultiLabels(), trainPredict)+ "\t");
            System.out.print("trainOver: "+ Overlap.overlap(trainSet.getMultiLabels(), trainPredict)+ "\t");
            System.out.print("test Hamming loss : " + HammingLoss.hammingLoss(bmmClassifier, testSet) + "\t");
            System.out.print("testAcc  : " + Accuracy.accuracy(testSet.getMultiLabels(), testPredict) + "\t");
            System.out.print("testOver : " + Overlap.overlap(testSet.getMultiLabels(), testPredict) + "\t");
        }
    }


    private static void test4_powerset() throws Exception{
        System.out.println("powerset");
        MultiLabelClfDataSet trainSet = TRECFormat.loadMultiLabelClfDataSet(new File(TMP, "train.trec"), DataSetType.ML_CLF_DENSE, true);
        MultiLabelClfDataSet testSet = TRECFormat.loadMultiLabelClfDataSet(new File(TMP, "test.trec"), DataSetType.ML_CLF_DENSE, true);

        LPClassifier classifier = new LPClassifier(trainSet);
        LPOptimizer optimizer = new LPOptimizer(classifier,trainSet);
        Config config1 = new Config();
        config1.setString("classifier","lkboost");
        config1.setDouble("l1Ratio", 0);
        config1.setDouble("regularization", 0.00001);
        config1.setInt("numIters",200);
        optimizer.optimize(config1);

        MultiLabel[] trainPredict;
        MultiLabel[] testPredict;
        trainPredict = classifier.predict(trainSet);
        testPredict = classifier.predict(testSet);
        System.out.print("train Hamming loss : " + HammingLoss.hammingLoss(classifier, trainSet) + "\t");
        System.out.print("trainAcc : "+ Accuracy.accuracy(trainSet.getMultiLabels(), trainPredict)+ "\t");
        System.out.print("trainOver: "+ Overlap.overlap(trainSet.getMultiLabels(), trainPredict)+ "\t");
        System.out.print("test Hamming loss : " + HammingLoss.hammingLoss(classifier, testSet) + "\t");
        System.out.print("testAcc  : " + Accuracy.accuracy(testSet.getMultiLabels(), testPredict) + "\t");
        System.out.println("testOver : "+ Overlap.overlap(testSet.getMultiLabels(), testPredict)+ "\t");
    }


    private static void test4_mix() throws Exception{
        System.out.println("mix");
        MultiLabelClfDataSet trainSet = TRECFormat.loadMultiLabelClfDataSet(new File(DATASETS, "simulation/multi-label/flip_one/2_labels/train.trec"), DataSetType.ML_CLF_DENSE, true);
        MultiLabelClfDataSet testSet = TRECFormat.loadMultiLabelClfDataSet(new File(DATASETS, "simulation/multi-label/flip_one/2_labels/test.trec"), DataSetType.ML_CLF_DENSE, true);
        int numClusters = 3;
        double softmaxVariance = 100;
        double logitVariance = 100;
        BMMClassifier bmmClassifier = new BMMClassifier(trainSet.getNumClasses(),numClusters,trainSet.getNumFeatures());
        BMMOptimizer optimizer = new BMMOptimizer(bmmClassifier, trainSet,softmaxVariance,logitVariance);
        bmmClassifier.setAllowEmpty(true);
        bmmClassifier.setPredictMode("dynamic");
        BMMInitializer.initialize(bmmClassifier, trainSet, softmaxVariance, logitVariance);
        for (int i=1;i<=20;i++){
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
        System.out.println(bmmClassifier);
    }

    private static void test4_crf() throws Exception{
        System.out.println("crf");
        MultiLabelClfDataSet trainSet = TRECFormat.loadMultiLabelClfDataSet(new File(DATASETS, "simulation/multi-label/flip_one/5_labels/train.trec"), DataSetType.ML_CLF_DENSE, true);
        MultiLabelClfDataSet testSet = TRECFormat.loadMultiLabelClfDataSet(new File(DATASETS, "simulation/multi-label/flip_one/5_labels/test.trec"), DataSetType.ML_CLF_DENSE, true);

        MLLogisticTrainer trainer = MLLogisticTrainer.getBuilder().setGaussianPriorVariance(100).build();
        MLLogisticRegression classifier = trainer.train(trainSet);

        MultiLabel[] trainPredict;
        MultiLabel[] testPredict;
        trainPredict = classifier.predict(trainSet);
        testPredict = classifier.predict(testSet);
        System.out.print("train Hamming loss : " + HammingLoss.hammingLoss(classifier, trainSet) + "\t");
        System.out.print("trainAcc : "+ Accuracy.accuracy(trainSet.getMultiLabels(), trainPredict)+ "\t");
        System.out.print("trainOver: "+ Overlap.overlap(trainSet.getMultiLabels(), trainPredict)+ "\t");
        System.out.print("test Hamming loss : " + HammingLoss.hammingLoss(classifier, testSet) + "\t");
        System.out.print("testAcc  : " + Accuracy.accuracy(testSet.getMultiLabels(), testPredict) + "\t");
        System.out.println("testOver : "+ Overlap.overlap(testSet.getMultiLabels(), testPredict)+ "\t");
    }


    private static void test5Dump() {
        MultiLabelClfDataSet all = MultiLabelSynthesizer.flipTwo(10000,2,5);
        List<Integer> trainIndices = IntStream.range(0,5000).mapToObj(i -> i).collect(Collectors.toList());
        List<Integer> testIndices = IntStream.range(5000,10000).mapToObj(i->i).collect(Collectors.toList());
        DataSetUtil.sampleData(all,trainIndices);
        MultiLabelClfDataSet trainSet = DataSetUtil.sampleData(all, trainIndices);
        MultiLabelClfDataSet testSet =  DataSetUtil.sampleData(all, testIndices);
        TRECFormat.save(trainSet, new File(TMP,"train.trec"));
        TRECFormat.save(testSet, new File(TMP,"test.trec"));
    }


    private static void test5_br() throws Exception{
        System.out.println("binary");
        MultiLabelClfDataSet trainSet = TRECFormat.loadMultiLabelClfDataSet(new File(TMP,"train.trec"), DataSetType.ML_CLF_DENSE,true);
        MultiLabelClfDataSet testSet = TRECFormat.loadMultiLabelClfDataSet(new File(TMP, "test.trec"), DataSetType.ML_CLF_DENSE, true);
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
            System.out.print("train Hamming loss : " + HammingLoss.hammingLoss(bmmClassifier, trainSet) + "\t");
            System.out.print("trainAcc : "+ Accuracy.accuracy(trainSet.getMultiLabels(), trainPredict)+ "\t");
            System.out.print("trainOver: "+ Overlap.overlap(trainSet.getMultiLabels(), trainPredict)+ "\t");
            System.out.print("test Hamming loss : " + HammingLoss.hammingLoss(bmmClassifier, testSet) + "\t");
            System.out.print("testAcc  : " + Accuracy.accuracy(testSet.getMultiLabels(), testPredict) + "\t");
            System.out.print("testOver : " + Overlap.overlap(testSet.getMultiLabels(), testPredict) + "\t");
        }
    }


    private static void test5_powerset() throws Exception{
        System.out.println("powerset");
        MultiLabelClfDataSet trainSet = TRECFormat.loadMultiLabelClfDataSet(new File(TMP, "train.trec"), DataSetType.ML_CLF_DENSE, true);
        MultiLabelClfDataSet testSet = TRECFormat.loadMultiLabelClfDataSet(new File(TMP, "test.trec"), DataSetType.ML_CLF_DENSE, true);

        LPClassifier classifier = new LPClassifier(trainSet);
        LPOptimizer optimizer = new LPOptimizer(classifier,trainSet);
        Config config1 = new Config();
        config1.setString("classifier","logistic");
        config1.setDouble("l1Ratio", 0);
        config1.setDouble("regularization", 0.00001);
        config1.setInt("numIters",200);
        optimizer.optimize(config1);

        MultiLabel[] trainPredict;
        MultiLabel[] testPredict;
        trainPredict = classifier.predict(trainSet);
        testPredict = classifier.predict(testSet);
        System.out.print("train Hamming loss : " + HammingLoss.hammingLoss(classifier, trainSet) + "\t");
        System.out.print("trainAcc : "+ Accuracy.accuracy(trainSet.getMultiLabels(), trainPredict)+ "\t");
        System.out.print("trainOver: "+ Overlap.overlap(trainSet.getMultiLabels(), trainPredict)+ "\t");
        System.out.print("test Hamming loss : " + HammingLoss.hammingLoss(classifier, testSet) + "\t");
        System.out.print("testAcc  : " + Accuracy.accuracy(testSet.getMultiLabels(), testPredict) + "\t");
        System.out.println("testOver : "+ Overlap.overlap(testSet.getMultiLabels(), testPredict)+ "\t");
    }


    private static void test5_mix() throws Exception{
        System.out.println("mix");
        MultiLabelClfDataSet trainSet = TRECFormat.loadMultiLabelClfDataSet(new File(TMP, "train.trec"), DataSetType.ML_CLF_DENSE, true);
        MultiLabelClfDataSet testSet = TRECFormat.loadMultiLabelClfDataSet(new File(TMP, "test.trec"), DataSetType.ML_CLF_DENSE, true);
        int numClusters = 20;
        double softmaxVariance = 100;
        double logitVariance = 100;
        BMMClassifier bmmClassifier = new BMMClassifier(trainSet.getNumClasses(),numClusters,trainSet.getNumFeatures());
        BMMOptimizer optimizer = new BMMOptimizer(bmmClassifier, trainSet,softmaxVariance,logitVariance);
        bmmClassifier.setAllowEmpty(true);
        bmmClassifier.setPredictMode("dynamic");
        BMMInitializer.initialize(bmmClassifier, trainSet, softmaxVariance, logitVariance);
        for (int i=1;i<=20;i++){
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


    private static void test6Dump() {
        MultiLabelClfDataSet all = MultiLabelSynthesizer.flipOneNonUniform(10000);
        List<Integer> trainIndices = IntStream.range(0,5000).mapToObj(i -> i).collect(Collectors.toList());
        List<Integer> testIndices = IntStream.range(5000,10000).mapToObj(i->i).collect(Collectors.toList());
        DataSetUtil.sampleData(all,trainIndices);
        MultiLabelClfDataSet trainSet = DataSetUtil.sampleData(all, trainIndices);
        MultiLabelClfDataSet testSet =  DataSetUtil.sampleData(all, testIndices);
        TRECFormat.save(trainSet, new File(TMP,"train.trec"));
        TRECFormat.save(testSet, new File(TMP,"test.trec"));
    }


    private static void test6_br() throws Exception{
        System.out.println("binary");
        MultiLabelClfDataSet trainSet = TRECFormat.loadMultiLabelClfDataSet(new File(DATASETS,"simulation/multi-label/flip_one/4_labels_nonuniform/train.trec"), DataSetType.ML_CLF_DENSE,true);
        MultiLabelClfDataSet testSet = TRECFormat.loadMultiLabelClfDataSet(new File(DATASETS, "simulation/multi-label/flip_one/4_labels_nonuniform/test.trec"), DataSetType.ML_CLF_DENSE, true);

//        MultiLabelClfDataSet trainSet = TRECFormat.loadMultiLabelClfDataSet(new File(TMP,"train.trec"), DataSetType.ML_CLF_DENSE,true);
//        MultiLabelClfDataSet testSet = TRECFormat.loadMultiLabelClfDataSet(new File(TMP, "test.trec"), DataSetType.ML_CLF_DENSE, true);

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
            System.out.print("train Hamming loss : " + HammingLoss.hammingLoss(bmmClassifier, trainSet) + "\t");
            System.out.print("trainAcc : "+ Accuracy.accuracy(trainSet.getMultiLabels(), trainPredict)+ "\t");
            System.out.print("trainOver: "+ Overlap.overlap(trainSet.getMultiLabels(), trainPredict)+ "\t");
            System.out.print("test Hamming loss : " + HammingLoss.hammingLoss(bmmClassifier, testSet) + "\t");
            System.out.print("testAcc  : " + Accuracy.accuracy(testSet.getMultiLabels(), testPredict) + "\t");
            System.out.print("testOver : " + Overlap.overlap(testSet.getMultiLabels(), testPredict) + "\t");
        }
    }


    private static void test6_powerset() throws Exception{
        System.out.println("powerset");
//        MultiLabelClfDataSet trainSet = TRECFormat.loadMultiLabelClfDataSet(new File(TMP,"train.trec"), DataSetType.ML_CLF_DENSE,true);
//        MultiLabelClfDataSet testSet = TRECFormat.loadMultiLabelClfDataSet(new File(TMP, "test.trec"), DataSetType.ML_CLF_DENSE, true);

        MultiLabelClfDataSet trainSet = TRECFormat.loadMultiLabelClfDataSet(new File(DATASETS, "simulation/multi-label/flip_one/4_labels_nonuniform/train.trec"), DataSetType.ML_CLF_DENSE, true);
        MultiLabelClfDataSet testSet = TRECFormat.loadMultiLabelClfDataSet(new File(DATASETS, "simulation/multi-label/flip_one/4_labels_nonuniform/test.trec"), DataSetType.ML_CLF_DENSE, true);

        LPClassifier classifier = new LPClassifier(trainSet);
        LPOptimizer optimizer = new LPOptimizer(classifier,trainSet);
        Config config1 = new Config();
        config1.setString("classifier","logistic");
        config1.setDouble("l1Ratio", 0);
        config1.setDouble("regularization", 0.00001);
        config1.setInt("numIters",200);
        optimizer.optimize(config1);

        MultiLabel[] trainPredict;
        MultiLabel[] testPredict;
        trainPredict = classifier.predict(trainSet);
        testPredict = classifier.predict(testSet);
        System.out.print("train Hamming loss : " + HammingLoss.hammingLoss(classifier, trainSet) + "\t");
        System.out.print("trainAcc : "+ Accuracy.accuracy(trainSet.getMultiLabels(), trainPredict)+ "\t");
        System.out.print("trainOver: "+ Overlap.overlap(trainSet.getMultiLabels(), trainPredict)+ "\t");
        System.out.print("test Hamming loss : " + HammingLoss.hammingLoss(classifier, testSet) + "\t");
        System.out.print("testAcc  : " + Accuracy.accuracy(testSet.getMultiLabels(), testPredict) + "\t");
        System.out.println("testOver : "+ Overlap.overlap(testSet.getMultiLabels(), testPredict)+ "\t");
    }


    private static void test6_mix() throws Exception{
        System.out.println("mix");
        MultiLabelClfDataSet trainSet = TRECFormat.loadMultiLabelClfDataSet(new File(DATASETS, "simulation/multi-label/flip_one/4_labels_nonuniform/train.trec"), DataSetType.ML_CLF_DENSE, true);
        MultiLabelClfDataSet testSet = TRECFormat.loadMultiLabelClfDataSet(new File(DATASETS, "simulation/multi-label/flip_one/4_labels_nonuniform/test.trec"), DataSetType.ML_CLF_DENSE, true);
        int numClusters = 5;
        double softmaxVariance = 100;
        double logitVariance = 100;
        BMMClassifier bmmClassifier = new BMMClassifier(trainSet.getNumClasses(),numClusters,trainSet.getNumFeatures());
        BMMOptimizer optimizer = new BMMOptimizer(bmmClassifier, trainSet,softmaxVariance,logitVariance);
        bmmClassifier.setAllowEmpty(true);
        bmmClassifier.setPredictMode("dynamic");
        BMMInitializer.initialize(bmmClassifier, trainSet, softmaxVariance, logitVariance);
        for (int i=1;i<=20;i++){
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
        System.out.println(bmmClassifier);
    }

    private static void test6_mix_boost() throws Exception{
        System.out.println("mix");
        MultiLabelClfDataSet trainSet = TRECFormat.loadMultiLabelClfDataSet(new File(DATASETS, "simulation/multi-label/flip_one/4_labels_nonuniform/train.trec"), DataSetType.ML_CLF_DENSE, true);
        MultiLabelClfDataSet testSet = TRECFormat.loadMultiLabelClfDataSet(new File(DATASETS, "simulation/multi-label/flip_one/4_labels_nonuniform/test.trec"), DataSetType.ML_CLF_DENSE, true);
        int numClusters = 10;
        BMMClassifier bmmClassifier = BMMClassifier.newMixBoost(trainSet.getNumClasses(),numClusters,trainSet.getNumFeatures());
        MixBoostOptimizer optimizer = new MixBoostOptimizer(bmmClassifier,trainSet);
        bmmClassifier.setAllowEmpty(true);
        bmmClassifier.setPredictMode("dynamic");
        MixBoostInitializer.initialize(bmmClassifier, trainSet);
        for (int i=1;i<=100;i++){
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