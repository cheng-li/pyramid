package edu.neu.ccs.pyramid.core.multilabel_classification.crf;

import edu.neu.ccs.pyramid.core.configuration.Config;
import edu.neu.ccs.pyramid.core.dataset.DataSetType;
import edu.neu.ccs.pyramid.core.dataset.MultiLabel;
import edu.neu.ccs.pyramid.core.dataset.TRECFormat;
import edu.neu.ccs.pyramid.core.eval.Accuracy;
import edu.neu.ccs.pyramid.core.eval.Overlap;
import edu.neu.ccs.pyramid.core.optimization.GradientDescent;
import edu.neu.ccs.pyramid.core.dataset.MultiLabelClfDataSet;
import edu.neu.ccs.pyramid.core.optimization.LBFGS;

import java.io.File;

/**
 * Created by Rainicy on 12/14/15.
 */
public class CMLCRFTest {
    private static final Config config = new Config("config/local.properties");
    private static final String DATASETS = config.getString("input.datasets");
    private static final String TMP = config.getString("output.tmp");

    public static void main(String[] args) throws Exception{
        test1();
//        test2();

//        test3();
//        test4();
//        test5();
//        test6();
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
                    System.out.print("\tTrain acc: " + String.format("%.4f", Accuracy.accuracy(trainSet.getMultiLabels(), predTrain)));
                    System.out.print("\tTrain overlap " + String.format("%.4f", Overlap.overlap(trainSet.getMultiLabels(), predTrain)));
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
}
