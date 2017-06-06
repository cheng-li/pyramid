package edu.neu.ccs.pyramid.core.classification.logistic_regression;

import edu.neu.ccs.pyramid.core.configuration.Config;
import edu.neu.ccs.pyramid.core.dataset.TRECFormat;
import edu.neu.ccs.pyramid.core.eval.Accuracy;
import edu.neu.ccs.pyramid.core.optimization.LBFGS;
import edu.neu.ccs.pyramid.core.optimization.Optimizable;
import edu.neu.ccs.pyramid.core.optimization.Terminator;
import edu.neu.ccs.pyramid.core.dataset.ClfDataSet;
import edu.neu.ccs.pyramid.core.dataset.DataSetType;

import java.io.File;

public class RidgeLogisticOptimizerTest {
    private static final Config config = new Config("config/local.properties");
    private static final String DATASETS = config.getString("input.datasets");
    private static final String TMP = config.getString("output.tmp");
    public static void main(String[] args) throws Exception{
//        test1();
//        test2();
        test3();
    }

    private static void test1() throws Exception{
//        ClfDataSet dataSet = TRECFormat.loadClfDataSet(new File(DATASETS, "/imdb/3/train.trec"),
//                DataSetType.CLF_SPARSE, true);
//        ClfDataSet testSet = TRECFormat.loadClfDataSet(new File(DATASETS, "/imdb/3/test.trec"),
//                DataSetType.CLF_SPARSE, true);
        ClfDataSet dataSet = TRECFormat.loadClfDataSet(new File(DATASETS, "20newsgroup/1/train.trec"),
                DataSetType.CLF_SPARSE, true);
        ClfDataSet testSet = TRECFormat.loadClfDataSet(new File(DATASETS, "20newsgroup/1/test.trec"),
                DataSetType.CLF_SPARSE, true);
//        ClfDataSet dataSet = TRECFormat.loadClfDataSet(new File(DATASETS, "/spam/trec_data/train.trec"),
//                DataSetType.CLF_SPARSE, true);
//        ClfDataSet testSet = TRECFormat.loadClfDataSet(new File(DATASETS, "/spam/trec_data/test.trec"),
//                DataSetType.CLF_SPARSE, true);
        double variance =1000;
        LogisticRegression logisticRegression = new LogisticRegression(dataSet.getNumClasses(),dataSet.getNumFeatures());
        RidgeLogisticOptimizer optimizer = new RidgeLogisticOptimizer(logisticRegression,dataSet,variance, true);
        optimizer.getOptimizer().getTerminator().setMaxIteration(10000).setMode(Terminator.Mode.STANDARD);
        System.out.println("after initialization");
        System.out.println("train acc = " + Accuracy.accuracy(logisticRegression, dataSet));
        System.out.println("test acc = "+Accuracy.accuracy(logisticRegression,testSet));
        optimizer.optimize();
        System.out.println("after training");
        System.out.println("train acc = " + Accuracy.accuracy(logisticRegression, dataSet));
        System.out.println("test acc = "+Accuracy.accuracy(logisticRegression,testSet));
        System.out.println(optimizer.getOptimizer().getTerminator().getHistory());
        System.out.println(logisticRegression);
    }

    private static void test2() throws Exception{
        ClfDataSet dataSet = TRECFormat.loadClfDataSet(new File(DATASETS, "/spam/trec_data/train.trec"),
                DataSetType.CLF_SPARSE, true);
        ClfDataSet testSet = TRECFormat.loadClfDataSet(new File(DATASETS, "/spam/trec_data/test.trec"),
                DataSetType.CLF_SPARSE, true);
        LogisticRegression logisticRegression = new LogisticRegression(dataSet.getNumClasses(),dataSet.getNumFeatures());

        // generate equal weights
        double[] gammas = new double[dataSet.getNumDataPoints()];
        for (int n=0; n<dataSet.getNumDataPoints(); n++) {
            gammas[n] =1.0;
        }

        // generate the targets distributions.
        int[] labels = dataSet.getLabels();
        double[][] targets = new double[dataSet.getNumDataPoints()][2];
        for (int n=0; n<dataSet.getNumDataPoints(); n++) {
            int label = labels[n];
            if (label == 0.0) {
                targets[n][0] = 1;
            } else {
                targets[n][1] = 1;
            }
        }

        RidgeLogisticOptimizer optimizer = new RidgeLogisticOptimizer(logisticRegression,dataSet,gammas,targets,500, true);
        optimizer.getOptimizer().getTerminator().setMaxIteration(10000).setMode(Terminator.Mode.STANDARD);
        System.out.println("after initialization");
        System.out.println("train acc = " + Accuracy.accuracy(logisticRegression, dataSet));
        System.out.println("test acc = "+Accuracy.accuracy(logisticRegression,testSet));
        optimizer.optimize();
        System.out.println("after training");
        System.out.println("train acc = " + Accuracy.accuracy(logisticRegression, dataSet));
        System.out.println("test acc = "+Accuracy.accuracy(logisticRegression,testSet));
        System.out.println(optimizer.getOptimizer().getTerminator().getHistory());
    }


    private static void test3() throws Exception{
//        ClfDataSet dataSet = TRECFormat.loadClfDataSet(new File(DATASETS, "/imdb/3/train.trec"),
//                DataSetType.CLF_SPARSE, true);
//        ClfDataSet testSet = TRECFormat.loadClfDataSet(new File(DATASETS, "/imdb/3/test.trec"),
//                DataSetType.CLF_SPARSE, true);
        ClfDataSet dataSet = TRECFormat.loadClfDataSet(new File(DATASETS, "20newsgroup/1/train.trec"),
                DataSetType.CLF_SPARSE, true);
        ClfDataSet testSet = TRECFormat.loadClfDataSet(new File(DATASETS, "20newsgroup/1/test.trec"),
                DataSetType.CLF_SPARSE, true);
//        ClfDataSet dataSet = TRECFormat.loadClfDataSet(new File(DATASETS, "/spam/trec_data/train.trec"),
//                DataSetType.CLF_SPARSE, true);
//        ClfDataSet testSet = TRECFormat.loadClfDataSet(new File(DATASETS, "/spam/trec_data/test.trec"),
//                DataSetType.CLF_SPARSE, true);
        double variance =1000;
        LogisticRegression logisticRegression = new LogisticRegression(dataSet.getNumClasses(),dataSet.getNumFeatures());
        Optimizable.ByGradientValue loss = new LogisticLoss(logisticRegression,dataSet, variance, true);
//        GradientDescent optimizer = new GradientDescent(loss);
        LBFGS optimizer = new LBFGS(loss);
        System.out.println("after initialization");
        System.out.println("train acc = " + Accuracy.accuracy(logisticRegression, dataSet));
        System.out.println("test acc = "+Accuracy.accuracy(logisticRegression,testSet));
        for (int i=0;i<200;i++){
            optimizer.iterate();
            System.out.println("after iteration "+i);
            System.out.println("loss = "+loss.getValue());

            System.out.println("train acc = " + Accuracy.accuracy(logisticRegression, dataSet));
            System.out.println("test acc = "+Accuracy.accuracy(logisticRegression,testSet));
//            System.out.println(logisticRegression);
        }

    }
}