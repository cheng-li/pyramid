package edu.neu.ccs.pyramid.multilabel_classification.crf;

import edu.neu.ccs.pyramid.configuration.Config;
import edu.neu.ccs.pyramid.dataset.DataSetType;
import edu.neu.ccs.pyramid.dataset.MultiLabel;
import edu.neu.ccs.pyramid.dataset.MultiLabelClfDataSet;
import edu.neu.ccs.pyramid.dataset.TRECFormat;
import edu.neu.ccs.pyramid.eval.Accuracy;
import edu.neu.ccs.pyramid.eval.FMeasure;
import edu.neu.ccs.pyramid.eval.MLMeasures;
import edu.neu.ccs.pyramid.eval.Overlap;
import edu.neu.ccs.pyramid.multilabel_classification.crf.CMLCRF;
import edu.neu.ccs.pyramid.multilabel_classification.crf.CRFLoss;
import edu.neu.ccs.pyramid.optimization.GradientDescent;
import edu.neu.ccs.pyramid.optimization.LBFGS;
import edu.neu.ccs.pyramid.optimization.Optimizer;
import edu.neu.ccs.pyramid.simulation.MultiLabelSynthesizer;

import java.io.File;
import java.util.Arrays;
import java.util.concurrent.TimeUnit;

/**
 * Created by Rainicy on 12/14/15.
 */
public class CMLCRFTest {
    private static final Config config = new Config("config/local.properties");
    private static final String DATASETS = config.getString("input.datasets");
    private static final String TMP = config.getString("output.tmp");

    public static void main(String[] args) throws Exception{
//        test1();
//        test2();

//        test3();
//        test4();
//        test5();
//        test6();

//        test7();
        test8();
    }

    public static void test2() throws Exception {
        System.out.println(config);

        MultiLabelClfDataSet trainSet = TRECFormat.loadMultiLabelClfDataSet(config.getString("input.trainData"),
                DataSetType.ML_CLF_DENSE, true);
        MultiLabelClfDataSet testSet = TRECFormat.loadMultiLabelClfDataSet(config.getString("input.testData"),
                DataSetType.ML_CLF_DENSE, true);
        double gaussianVariance = config.getDouble("gaussianVariance");

        // loading or save model infos.
        String output = config.getString("output");
        String modelName = config.getString("modelName");

        CMLCRF cmlcrf;
        MultiLabel[] predTrain;
        MultiLabel[] predTest;

        if (config.getBoolean("train.warmStart")) {
            cmlcrf = CMLCRF.deserialize(new File(output, modelName));
            System.out.println("loading model:");
            System.out.println(cmlcrf);
        } else{
            cmlcrf = new CMLCRF(trainSet);
            CRFLoss crfLoss = new CRFLoss(cmlcrf, trainSet, gaussianVariance);

            if (config.getBoolean("isLBFGS")) {
                LBFGS optimizer = new LBFGS(crfLoss);
                optimizer.getTerminator().setAbsoluteEpsilon(0.1);

                for (int i=0; i<config.getInt("numRounds"); i++) {
                    optimizer.iterate();
                    predTrain = cmlcrf.predict(trainSet);
                    predTest = cmlcrf.predict(testSet);
                    System.out.print("iter: "+ String.format("%04d", i));
                    System.out.print("\tTrain acc: " + String.format("%.4f",Accuracy.accuracy(trainSet.getMultiLabels(), predTrain)));
                    System.out.print("\tTrain overlap " + String.format("%.4f",Overlap.overlap(trainSet.getMultiLabels(), predTrain)));
                    System.out.print("\tTest acc: " + String.format("%.4f",Accuracy.accuracy(testSet.getMultiLabels(), predTest)));
                    System.out.println("\tTest overlap " + String.format("%.4f",Overlap.overlap(testSet.getMultiLabels(), predTest)));
                }


            } else {
                GradientDescent optimizer = new GradientDescent(crfLoss);
                for (int i=0; i<config.getInt("numRounds"); i++) {
                    optimizer.iterate();
                    predTrain = cmlcrf.predict(trainSet);
                    predTest = cmlcrf.predict(testSet);
                    System.out.print("iter: "+ String.format("%04d", i));
                    System.out.print("\tTrain acc: " + String.format("%.4f",Accuracy.accuracy(trainSet.getMultiLabels(), predTrain)));
                    System.out.print("\tTrain overlap " + String.format("%.4f",Overlap.overlap(trainSet.getMultiLabels(), predTrain)));
                    System.out.print("\tTest acc: " + String.format("%.4f",Accuracy.accuracy(testSet.getMultiLabels(), predTest)));
                    System.out.println("\tTest overlap " + String.format("%.4f",Overlap.overlap(testSet.getMultiLabels(), predTest)));
                }
            }
        }

        System.out.println();
        System.out.println();
        System.out.println("--------------------------------Results-----------------------------\n");
        predTrain = cmlcrf.predict(trainSet);
        predTest = cmlcrf.predict(testSet);
        System.out.print("Train acc: " + String.format("%.4f",Accuracy.accuracy(trainSet.getMultiLabels(), predTrain)));
        System.out.print("\tTrain overlap " + String.format("%.4f",Overlap.overlap(trainSet.getMultiLabels(), predTrain)));
        System.out.print("\tTest acc: " + String.format("%.4f",Accuracy.accuracy(testSet.getMultiLabels(), predTest)));
        System.out.println("\tTest overlap " + String.format("%.4f",Overlap.overlap(testSet.getMultiLabels(), predTest)));

        if (config.getBoolean("saveModel")) {
            (new File(output)).mkdirs();
            File serializeModel = new File(output, modelName);
            cmlcrf.serialize(serializeModel);
        }
    }

    private static void test1() throws Exception{
        MultiLabelClfDataSet dataSet = TRECFormat.loadMultiLabelClfDataSet(new File(DATASETS, "spam/trec_data/train.trec"),
                DataSetType.ML_CLF_SPARSE, true);
        MultiLabelClfDataSet testSet = TRECFormat.loadMultiLabelClfDataSet(new File(DATASETS, "spam/trec_data/test.trec"),
                DataSetType.ML_CLF_SPARSE, true);

        CMLCRF cmlcrf = new CMLCRF(dataSet);
        CRFLoss crfLoss = new CRFLoss(cmlcrf,dataSet,1);
        cmlcrf.setConsiderPair(true);


        MultiLabel[] predTrain;
        MultiLabel[] predTest;

        LBFGS optimizer = new LBFGS(crfLoss);
        for (int i=0; i<5000; i++) {

//            System.out.print("Obj: " + optimizer.getTerminator().getLastValue());
            System.out.println("iter: "+ i);
            optimizer.iterate();
            System.out.println(crfLoss.getValue());
            predTrain = cmlcrf.predict(dataSet);
            predTest = cmlcrf.predict(testSet);
            System.out.print("\tTrain acc: " + Accuracy.accuracy(dataSet.getMultiLabels(), predTrain));
            System.out.print("\tTrain overlap " + Overlap.overlap(dataSet.getMultiLabels(), predTrain));
            System.out.print("\tTest acc: " + Accuracy.accuracy(testSet.getMultiLabels(), predTest));
            System.out.println("\tTest overlap " + Overlap.overlap(testSet.getMultiLabels(), predTest));
//            System.out.println("crf = "+cmlcrf.getWeights());
//            System.out.println(Arrays.toString(predTrain));
        }



//        LBFGS optimizer = new LBFGS(crfLoss);
//        optimizer.getTerminator().setAbsoluteEpsilon(0.01);
//        optimizer.optimize();
//        predTrain = cmlcrf.predict(dataSet);
//        predTest = cmlcrf.predict(testSet);
//        System.out.print("Train acc: " + Accuracy.accuracy(dataSet.getMultiLabels(), predTrain));
//        System.out.print("\tTrain overlap " + Overlap.overlap(dataSet.getMultiLabels(), predTrain));
//        System.out.print("\tTest acc: " + Accuracy.accuracy(testSet.getMultiLabels(), predTest));
//        System.out.println("\tTest overlap " + Overlap.overlap(testSet.getMultiLabels(), predTest));

    }

    private static void test3() throws Exception{
        MultiLabelClfDataSet dataSet = TRECFormat.loadMultiLabelClfDataSet(new File(DATASETS, "/imdb/3/train.trec"),
                DataSetType.ML_CLF_SPARSE, true);
        MultiLabelClfDataSet testSet = TRECFormat.loadMultiLabelClfDataSet(new File(DATASETS, "/imdb/3/test.trec"),
                DataSetType.ML_CLF_SPARSE, true);

        CMLCRF cmlcrf = new CMLCRF(dataSet);
        CRFLoss crfLoss = new CRFLoss(cmlcrf,dataSet,1);


        MultiLabel[] predTrain;
        MultiLabel[] predTest;

        LBFGS optimizer = new LBFGS(crfLoss);
        for (int i=0; i<50; i++) {

//            System.out.print("Obj: " + optimizer.getTerminator().getLastValue());
            System.out.println("iter: "+ i);
            optimizer.iterate();
            System.out.println(crfLoss.getValue());
            predTrain = cmlcrf.predict(dataSet);
            predTest = cmlcrf.predict(testSet);
            System.out.print("\tTrain acc: " + Accuracy.accuracy(dataSet.getMultiLabels(), predTrain));
            System.out.print("\tTrain overlap " + Overlap.overlap(dataSet.getMultiLabels(), predTrain));
            System.out.print("\tTest acc: " + Accuracy.accuracy(testSet.getMultiLabels(), predTest));
            System.out.println("\tTest overlap " + Overlap.overlap(testSet.getMultiLabels(), predTest));
//            System.out.println("crf = "+cmlcrf.getWeights());
//            System.out.println(Arrays.toString(predTrain));
        }



    }

    private static void test4() throws Exception{
        MultiLabelClfDataSet dataSet = TRECFormat.loadMultiLabelClfDataSet(new File(DATASETS, "20newsgroup/1/train.trec"),
                DataSetType.ML_CLF_SPARSE, true);
        MultiLabelClfDataSet testSet = TRECFormat.loadMultiLabelClfDataSet(new File(DATASETS, "20newsgroup/1/test.trec"),
                DataSetType.ML_CLF_SPARSE, true);

        CMLCRF cmlcrf = new CMLCRF(dataSet);
        CRFLoss crfLoss = new CRFLoss(cmlcrf,dataSet,1);


        MultiLabel[] predTrain;
        MultiLabel[] predTest;

        LBFGS optimizer = new LBFGS(crfLoss);
        for (int i=0; i<50; i++) {

//            System.out.print("Obj: " + optimizer.getTerminator().getLastValue());
            System.out.println("iter: "+ i);
            optimizer.iterate();
            System.out.println(crfLoss.getValue());
            predTrain = cmlcrf.predict(dataSet);
            predTest = cmlcrf.predict(testSet);
            System.out.print("\tTrain acc: " + Accuracy.accuracy(dataSet.getMultiLabels(), predTrain));
            System.out.print("\tTrain overlap " + Overlap.overlap(dataSet.getMultiLabels(), predTrain));
            System.out.print("\tTest acc: " + Accuracy.accuracy(testSet.getMultiLabels(), predTest));
            System.out.println("\tTest overlap " + Overlap.overlap(testSet.getMultiLabels(), predTest));
//            System.out.println("crf = "+cmlcrf.getWeights());
//            System.out.println(Arrays.toString(predTrain));
        }



    }

    private static void test5() throws Exception{
        MultiLabelClfDataSet dataSet = TRECFormat.loadMultiLabelClfDataSet(new File(DATASETS, "ohsumed/3/train.trec"),
                DataSetType.ML_CLF_SPARSE, true);
        MultiLabelClfDataSet testSet = TRECFormat.loadMultiLabelClfDataSet(new File(DATASETS, "ohsumed/3/test.trec"),
                DataSetType.ML_CLF_SPARSE, true);

        CMLCRF cmlcrf = new CMLCRF(dataSet);
        CRFLoss crfLoss = new CRFLoss(cmlcrf,dataSet,1);
        cmlcrf.setConsiderPair(false);


        MultiLabel[] predTrain;
        MultiLabel[] predTest;

        LBFGS optimizer = new LBFGS(crfLoss);
        for (int i=0; i<5; i++) {
//            System.out.print("Obj: " + optimizer.getTerminator().getLastValue());
            System.out.println("iter: "+ i);
            optimizer.iterate();
            System.out.println(crfLoss.getValue());
            predTrain = cmlcrf.predict(dataSet);
            predTest = cmlcrf.predict(testSet);
            System.out.print("\tTrain acc: " + Accuracy.accuracy(dataSet.getMultiLabels(), predTrain));
            System.out.print("\tTrain overlap " + Overlap.overlap(dataSet.getMultiLabels(), predTrain));
            System.out.print("\tTest acc: " + Accuracy.accuracy(testSet.getMultiLabels(), predTest));
            System.out.println("\tTest overlap " + Overlap.overlap(testSet.getMultiLabels(), predTest));
//            System.out.println("crf = "+cmlcrf.getWeights());
//            System.out.println(Arrays.toString(predTrain));
        }

        CRFLoss crfLoss2 = new CRFLoss(cmlcrf,dataSet,1);
        cmlcrf.setConsiderPair(true);
        LBFGS optimizer2 = new LBFGS(crfLoss2);
        for (int i=0; i<50; i++) {
            System.out.println("consider pairs");
//            System.out.print("Obj: " + optimizer.getTerminator().getLastValue());
            System.out.println("iter: "+ i);
            optimizer2.iterate();
            System.out.println(crfLoss2.getValue());
            predTrain = cmlcrf.predict(dataSet);
            predTest = cmlcrf.predict(testSet);
            System.out.print("\tTrain acc: " + Accuracy.accuracy(dataSet.getMultiLabels(), predTrain));
            System.out.print("\tTrain overlap " + Overlap.overlap(dataSet.getMultiLabels(), predTrain));
            System.out.print("\tTest acc: " + Accuracy.accuracy(testSet.getMultiLabels(), predTest));
            System.out.println("\tTest overlap " + Overlap.overlap(testSet.getMultiLabels(), predTest));
//            System.out.println("crf = "+cmlcrf.getWeights());
//            System.out.println(Arrays.toString(predTrain));
        }



    }

    private static void test6() throws Exception{
        MultiLabelClfDataSet dataSet = TRECFormat.loadMultiLabelClfDataSet(new File(DATASETS, "medical/train"),
                DataSetType.ML_CLF_SPARSE, true);
        MultiLabelClfDataSet testSet = TRECFormat.loadMultiLabelClfDataSet(new File(DATASETS, "medical/test"),
                DataSetType.ML_CLF_SPARSE, true);

        CMLCRF cmlcrf = new CMLCRF(dataSet);
        CRFLoss crfLoss = new CRFLoss(cmlcrf,dataSet,1);


        MultiLabel[] predTrain;
        MultiLabel[] predTest;

        LBFGS optimizer = new LBFGS(crfLoss);
        for (int i=0; i<50; i++) {

//            System.out.print("Obj: " + optimizer.getTerminator().getLastValue());
            System.out.println("iter: "+ i);
            optimizer.iterate();
            System.out.println(crfLoss.getValue());
            predTrain = cmlcrf.predict(dataSet);
            predTest = cmlcrf.predict(testSet);
            System.out.print("\tTrain acc: " + Accuracy.accuracy(dataSet.getMultiLabels(), predTrain));
            System.out.print("\tTrain overlap " + Overlap.overlap(dataSet.getMultiLabels(), predTrain));
            System.out.print("\tTest acc: " + Accuracy.accuracy(testSet.getMultiLabels(), predTest));
            System.out.println("\tTest overlap " + Overlap.overlap(testSet.getMultiLabels(), predTest));
//            System.out.println("crf = "+cmlcrf.getWeights());
//            System.out.println(Arrays.toString(predTrain));
        }
    }

    private static void test7() throws Exception {

        System.out.println(config);

        MultiLabelClfDataSet trainSet = TRECFormat.loadMultiLabelClfDataSet(config.getString("input.trainData"),
                DataSetType.ML_CLF_SEQ_SPARSE, true);
        MultiLabelClfDataSet testSet = TRECFormat.loadMultiLabelClfDataSet(config.getString("input.testData"),
                DataSetType.ML_CLF_SEQ_SPARSE, true);

        // loading or save model infos.
        String output = config.getString("output");
        String modelName = config.getString("modelName");

        CMLCRF cmlcrf = null;

        if (config.getString("train.warmStart").equals("true")) {
            cmlcrf = CMLCRF.deserialize(new File(output, modelName));
            System.out.println("loading model:");
            System.out.println(cmlcrf);
        } else if (config.getString("train.warmStart").equals("auto")) {
            cmlcrf = CMLCRF.deserialize(new File(output, modelName));
            System.out.println("retrain model:");
            CMLCRFElasticNet cmlcrfElasticNet = new CMLCRFElasticNet(cmlcrf, trainSet, config.getDouble("l1Ratio"), config.getDouble("regularization"));
            train(cmlcrfElasticNet, cmlcrf, trainSet, testSet, config);

        } else if (config.getString("train.warmStart").equals("false")) {
            cmlcrf = new CMLCRF(trainSet);
            cmlcrf.setConsiderPair(config.getBoolean("considerLabelPair"));
            CMLCRFElasticNet cmlcrfElasticNet = new CMLCRFElasticNet(cmlcrf, trainSet, config.getDouble("l1Ratio"), config.getDouble("regularization"));
            train(cmlcrfElasticNet, cmlcrf, trainSet, testSet, config);
        }

        System.out.println();
        System.out.println();
        System.out.println("--------------------------------Results-----------------------------\n");
        MLMeasures measures = new MLMeasures(cmlcrf, trainSet);
        System.out.println("========== Train ==========\n");
        System.out.println(measures);

        System.out.println("========== Test ==========\n");
        long startTimePred = System.nanoTime();
        MultiLabel[] preds = cmlcrf.predict(testSet);
        long stopTimePred = System.nanoTime();
        long predTime = stopTimePred - startTimePred;
        System.out.println("\nprediction time: " + TimeUnit.NANOSECONDS.toSeconds(predTime) + " sec.");
        System.out.println(new MLMeasures(cmlcrf, testSet));
        System.out.println("\n\n");
        InstanceF1Predictor pluginF1 = new InstanceF1Predictor(cmlcrf);
        System.out.println("Plugin F1");
        System.out.println(new MLMeasures(pluginF1, testSet));

        if (config.getBoolean("saveModel")) {
            (new File(output)).mkdirs();
            File serializeModel = new File(output, modelName);
            cmlcrf.serialize(serializeModel);
        }
    }
    private static void train(CMLCRFElasticNet optimizer, CMLCRF cmlcrf, MultiLabelClfDataSet trainSet, MultiLabelClfDataSet testSet, Config config) {
        MultiLabel[] predTrain;
        MultiLabel[] predTest;
        long startTime = System.nanoTime();
        for (int i=0; i < config.getInt("numRounds"); i++) {
            optimizer.iterate();
            predTrain = cmlcrf.predict(trainSet);
            predTest = cmlcrf.predict(testSet);
            System.out.print("iter: " + String.format("%04d", i));
            System.out.print("\tobjective: " + String.format("%.4f", optimizer.getValue()));
            System.out.print("\tTrain acc: " + String.format("%.4f", Accuracy.accuracy(trainSet.getMultiLabels(), predTrain)));
            System.out.print("\tTrain overlap " + String.format("%.4f", Overlap.overlap(trainSet.getMultiLabels(), predTrain)));
            System.out.print("\tTrain F1 " + String.format("%.4f", FMeasure.f1(trainSet.getMultiLabels(), predTrain)));
            System.out.print("\tTest acc: " + String.format("%.4f", Accuracy.accuracy(testSet.getMultiLabels(), predTest)));
            System.out.print("\tTest overlap " + String.format("%.4f", Overlap.overlap(testSet.getMultiLabels(), predTest)));
            System.out.println("\tTest F1 " + String.format("%.4f", FMeasure.f1(testSet.getMultiLabels(), predTest)));
        }

        long stopTime = System.nanoTime();
        long trainTime = stopTime - startTime;
        System.out.println("\ntraining time: " + TimeUnit.NANOSECONDS.toSeconds(trainTime) + " sec.");

    }

    private static void test8() throws Exception {

        System.out.println(config);

        MultiLabelClfDataSet trainSet = TRECFormat.loadMultiLabelClfDataSet(config.getString("input.trainData"),
                DataSetType.ML_CLF_SEQ_SPARSE, true);
        MultiLabelClfDataSet testSet = TRECFormat.loadMultiLabelClfDataSet(config.getString("input.testData"),
                DataSetType.ML_CLF_SEQ_SPARSE, true);

        // loading or save model infos.
        String output = config.getString("output");
        String modelName = config.getString("modelName");

        CMLCRF cmlcrf = new CMLCRF(trainSet);
        BlockwiseCD blockwiseCD = new BlockwiseCD(cmlcrf, trainSet, config.getDouble("l1Ratio"), config.getDouble("regularization"));

        MultiLabel[] predTrain;
        MultiLabel[] predTest;
        for (int i=0; i < 10000; i++) {
            blockwiseCD.iterate();
            predTrain = cmlcrf.predict(trainSet);
            predTest = cmlcrf.predict(testSet);
            System.out.print("iter: " + String.format("%04d", i));
            System.out.print("\tobjective: " + String.format("%.4f", blockwiseCD.getValue()));
            System.out.print("\tTrain acc: " + String.format("%.4f", Accuracy.accuracy(trainSet.getMultiLabels(), predTrain)));
            System.out.print("\tTrain overlap " + String.format("%.4f", Overlap.overlap(trainSet.getMultiLabels(), predTrain)));
            System.out.print("\tTrain F1 " + String.format("%.4f", FMeasure.f1(trainSet.getMultiLabels(), predTrain)));
            System.out.print("\tTest acc: " + String.format("%.4f", Accuracy.accuracy(testSet.getMultiLabels(), predTest)));
            System.out.print("\tTest overlap " + String.format("%.4f", Overlap.overlap(testSet.getMultiLabels(), predTest)));
            System.out.println("\tTest F1 " + String.format("%.4f", FMeasure.f1(testSet.getMultiLabels(), predTest)));
        }


        System.out.println();
        System.out.println();
        System.out.println("--------------------------------Results-----------------------------\n");
        MLMeasures measures = new MLMeasures(cmlcrf, trainSet);
        System.out.println("========== Train ==========\n");
        System.out.println(measures);

        System.out.println("========== Test ==========\n");
        long startTimePred = System.nanoTime();
        MultiLabel[] preds = cmlcrf.predict(testSet);
        long stopTimePred = System.nanoTime();
        long predTime = stopTimePred - startTimePred;
        System.out.println("\nprediction time: " + TimeUnit.NANOSECONDS.toSeconds(predTime) + " sec.");
        System.out.println(new MLMeasures(cmlcrf, testSet));
        System.out.println("\n\n");
        InstanceF1Predictor pluginF1 = new InstanceF1Predictor(cmlcrf);
        System.out.println("Plugin F1");
        System.out.println(new MLMeasures(pluginF1, testSet));

        if (config.getBoolean("saveModel")) {
            (new File(output)).mkdirs();
            File serializeModel = new File(output, modelName);
            cmlcrf.serialize(serializeModel);
        }
    }


    private static void test9(){
        MultiLabelClfDataSet train = MultiLabelSynthesizer.independentNoise();
        MultiLabelClfDataSet test = MultiLabelSynthesizer.independent();
        CMLCRF cmlcrf = new CMLCRF(train);

        cmlcrf.getWeights().getWeightsWithoutBiasForClass(0).set(0,0);
        cmlcrf.getWeights().getWeightsWithoutBiasForClass(0).set(1,1);

        cmlcrf.getWeights().getWeightsWithoutBiasForClass(1).set(0,1);
        cmlcrf.getWeights().getWeightsWithoutBiasForClass(1).set(1,1);

        cmlcrf.getWeights().getWeightsWithoutBiasForClass(2).set(0,1);
        cmlcrf.getWeights().getWeightsWithoutBiasForClass(2).set(1,0);

        cmlcrf.getWeights().getWeightsWithoutBiasForClass(3).set(0,1);
        cmlcrf.getWeights().getWeightsWithoutBiasForClass(3).set(1,-1);

        CRFLoss crfLoss = new CRFLoss(cmlcrf,train,1);

        System.out.println(cmlcrf);
        System.out.println("initial loss = "+crfLoss.getValue());
        System.out.println("training performance");
        System.out.println(new MLMeasures(cmlcrf, train));
        System.out.println("test performance");
        System.out.println(new MLMeasures(cmlcrf, test));


        LBFGS optimizer = new LBFGS(crfLoss);
        while (!optimizer.getTerminator().shouldTerminate()) {
            System.out.println("------------");
            optimizer.iterate();
            System.out.println(optimizer.getTerminator().getLastValue());
            System.out.println("training performance");
            System.out.println(new MLMeasures(cmlcrf, train));
            System.out.println("test performance");
            System.out.println(new MLMeasures(cmlcrf, test));
        }
        System.out.println(cmlcrf);
    }
}


