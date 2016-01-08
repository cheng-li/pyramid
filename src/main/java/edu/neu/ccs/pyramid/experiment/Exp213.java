//package edu.neu.ccs.pyramid.experiment;
//
//import edu.neu.ccs.pyramid.configuration.Config;
//import edu.neu.ccs.pyramid.dataset.DataSetType;
//import edu.neu.ccs.pyramid.dataset.MultiLabel;
//import edu.neu.ccs.pyramid.dataset.MultiLabelClfDataSet;
//import edu.neu.ccs.pyramid.dataset.TRECFormat;
//import edu.neu.ccs.pyramid.eval.Accuracy;
//import edu.neu.ccs.pyramid.eval.Overlap;
//import edu.neu.ccs.pyramid.multilabel_classification.bmm_variant.*;
//
//import java.io.File;
//
///**
// * label mixture model with boosting as classifier
// * Created by chengli on 11/10/15.
// */
//public class Exp213 {
//    public static void main(String[] args) throws Exception {
//        if (args.length != 1) {
//            throw new IllegalArgumentException("Please specify a properties file.");
//        }
//
//        Config config = new Config(args[0]);
//
//        System.out.println(config);
//
//        MultiLabelClfDataSet trainSet = TRECFormat.loadMultiLabelClfDataSet(config.getString("input.trainData"),
//                DataSetType.ML_CLF_SPARSE, true);
//        MultiLabelClfDataSet testSet = TRECFormat.loadMultiLabelClfDataSet(config.getString("input.testData"),
//                DataSetType.ML_CLF_SPARSE, true);
//
//        int numClusters = config.getInt("numClusters");
//        int numIterations = config.getInt("numIterations");
//
//
//        String output = config.getString("output");
//        String modelName = config.getString("modelName");
//
//        BMMClassifier bmmClassifier;
//        if (config.getBoolean("train.warmStart")) {
//            bmmClassifier = BMMClassifier.deserialize(new File(output, modelName));
//            bmmClassifier.setAllowEmpty(config.getBoolean("allowEmpty"));
//        } else {
//
//            bmmClassifier = BMMClassifier.getBuilder()
//                    .setNumClasses(trainSet.getNumClasses())
//                    .setNumFeatures(trainSet.getNumFeatures())
//                    .setNumClusters(numClusters)
//                    .setBinaryClassifierType("boost")
//                    .setMultiClassClassifierType("boost")
//                    .build();
//            bmmClassifier.setAllowEmpty(config.getBoolean("allowEmpty"));
//            MixBoostOptimizer optimizer = new MixBoostOptimizer(bmmClassifier,trainSet);
//            optimizer.setNumLeavesBinary(config.getInt("numLeavesBinary"));
//            optimizer.setNumLeavesMultiNomial(config.getInt("numLeavesMultiNomial"));
//            optimizer.setNumIterationsBinary(config.getInt("numIterationsBinary"));
//            optimizer.setNumIterationsMultiNomial(config.getInt("numIterationsMultiNomial"));
//            optimizer.setShrinkageBinary(config.getDouble("shrinkageBinary"));
//            optimizer.setShrinkageMultiNomial(config.getDouble("shrinkageMultiNomial"));
//            bmmClassifier.setPredictMode(config.getString("predictMode"));
//
//            MultiLabel[] trainPredict = bmmClassifier.predict(trainSet);
//            MultiLabel[] testPredict = bmmClassifier.predict(testSet);
//            System.out.print("random init" + "\t" );
//            System.out.print("objective: "+optimizer.getObjective()+ "\t");
//            System.out.print("trainAcc : "+ Accuracy.accuracy(trainSet.getMultiLabels(), trainPredict) + "\t");
//            System.out.print("trainOver: "+ Overlap.overlap(trainSet.getMultiLabels(), trainPredict) + "\t");
//            System.out.print("testACC  : "+ Accuracy.accuracy(testSet.getMultiLabels(),testPredict) + "\t");
//            System.out.println("testOver : "+ Overlap.overlap(testSet.getMultiLabels(), testPredict) + "\t");
//
//            MixBoostInitializer.initialize(bmmClassifier, trainSet);
//            System.out.println("after initialization");
//            trainPredict = bmmClassifier.predict(trainSet);
//            testPredict = bmmClassifier.predict(testSet);
//
//            System.out.print("objective: "+optimizer.getObjective()+ "\t");
//            System.out.print("trainAcc : " + Accuracy.accuracy(trainSet.getMultiLabels(), trainPredict) + "\t");
//            System.out.print("trainOver: " + Overlap.overlap(trainSet.getMultiLabels(), trainPredict) + "\t");
//            System.out.print("testACC  : " + Accuracy.accuracy(testSet.getMultiLabels(), testPredict) + "\t");
//            System.out.println("testOver : "+ Overlap.overlap(testSet.getMultiLabels(), testPredict) + "\t");
//
//            for (int i=1;i<=numIterations;i++){
//                optimizer.iterate();
//                trainPredict = bmmClassifier.predict(trainSet);
//                testPredict = bmmClassifier.predict(testSet);
//                System.out.print("iter : "+i + "\t");
//                System.out.print("objective: "+optimizer.getTerminator().getLastValue() + "\t");
//                System.out.print("trainAcc : "+ Accuracy.accuracy(trainSet.getMultiLabels(),trainPredict)+ "\t");
//                System.out.print("trainOver: "+ Overlap.overlap(trainSet.getMultiLabels(), trainPredict)+ "\t");
//                System.out.print("testAcc  : "+ Accuracy.accuracy(testSet.getMultiLabels(),testPredict)+ "\t");
//                System.out.println("testOver : "+ Overlap.overlap(testSet.getMultiLabels(), testPredict)+ "\t");
//            }
//            System.out.println("history = "+optimizer.getTerminator().getHistory());
//        }
//
//        System.out.println("--------------------------------Results-----------------------------\n");
//        System.out.println();
//        System.out.print("trainAcc : " + Accuracy.accuracy(bmmClassifier, trainSet) + "\t");
//        System.out.print("trainOver: "+ Overlap.overlap(bmmClassifier, trainSet)+ "\t");
//        System.out.print("testAcc  : "+ Accuracy.accuracy(bmmClassifier,testSet)+ "\t");
//        System.out.println("testOver : "+ Overlap.overlap(bmmClassifier, testSet)+ "\t");
//        System.out.println();
//        System.out.println();
//        System.out.println(bmmClassifier);
//
//        if (config.getBoolean("saveModel")) {
//            File serializeModel = new File(output, modelName);
//            bmmClassifier.serialize(serializeModel);
//        }
//    }
//}
