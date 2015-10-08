package edu.neu.ccs.pyramid.multilabel_classification.bmm;

import edu.neu.ccs.pyramid.configuration.Config;
import edu.neu.ccs.pyramid.dataset.ClfDataSet;
import edu.neu.ccs.pyramid.dataset.DataSetType;
import edu.neu.ccs.pyramid.dataset.MultiLabelClfDataSet;
import edu.neu.ccs.pyramid.dataset.TRECFormat;
import edu.neu.ccs.pyramid.eval.Accuracy;
import edu.neu.ccs.pyramid.eval.Overlap;
import edu.neu.ccs.pyramid.multilabel_classification.imlgb.IMLGradientBoosting;

import java.io.File;

import static org.junit.Assert.*;

public class BMMOptimizerTest {
    private static final Config config = new Config("/Users/Rainicy/Datasets/2.config");
    private static final String DATASETS = config.getString("input.datasets");
    private static final String TMP = config.getString("output.tmp");
    public static void main(String[] args) throws Exception{
        test1();
    }

    private static void test1() throws Exception{
        MultiLabelClfDataSet dataSet = TRECFormat.loadMultiLabelClfDataSet(new File(DATASETS, "/data_sets/train.trec"),
                DataSetType.ML_CLF_SPARSE, true);
        MultiLabelClfDataSet testSet = TRECFormat.loadMultiLabelClfDataSet(new File(DATASETS, "/data_sets/test.trec"),
                DataSetType.ML_CLF_SPARSE, true);
        BMMClassifier bmmClassifier = new BMMClassifier(dataSet.getNumClasses(),2,dataSet.getNumFeatures());
        BMMOptimizer optimizer = new BMMOptimizer(bmmClassifier,dataSet,10000);
        bmmClassifier.setNumSample(100);

        System.out.println("after initialization");
        System.out.println("train acc = "+ Accuracy.accuracy(bmmClassifier,dataSet));
        System.out.println("test acc = "+ Accuracy.accuracy(bmmClassifier,testSet));

        for (int i=1;i<=10;i++){
            optimizer.iterate();
            System.out.println("after iteration "+i);
            System.out.println("train acc = "+ Accuracy.accuracy(bmmClassifier,dataSet));
            System.out.println("train overlap = "+ Overlap.overlap(bmmClassifier, dataSet));
            System.out.println("test acc = "+ Accuracy.accuracy(bmmClassifier,testSet));
            System.out.println("test overlap = "+ Overlap.overlap(bmmClassifier, testSet));
        }


        System.out.println("history = "+optimizer.getTerminator().getHistory());
        System.out.println(bmmClassifier);
    }

    private static void test2() throws Exception{
        MultiLabelClfDataSet dataSet = TRECFormat.loadMultiLabelClfDataSet(new File(DATASETS, "ohsumed/3/train.trec"), DataSetType.ML_CLF_SPARSE, true);
        MultiLabelClfDataSet testSet = TRECFormat.loadMultiLabelClfDataSet(new File(DATASETS, "ohsumed/3/test.trec"), DataSetType.ML_CLF_SPARSE, true);

        int numClusters = 10;
        double variance = 10000;
        BMMClassifier bmmClassifier = new BMMClassifier(dataSet.getNumClasses(),numClusters,dataSet.getNumFeatures());
        BMMOptimizer optimizer = new BMMOptimizer(bmmClassifier,dataSet,variance);

        System.out.println("after initialization");
        System.out.println("train acc = "+ Accuracy.accuracy(bmmClassifier,dataSet));
        System.out.println("train overlap = "+ Overlap.overlap(bmmClassifier, dataSet));
        System.out.println("test acc = "+ Accuracy.accuracy(bmmClassifier,testSet));
        System.out.println("test overlap = "+ Overlap.overlap(bmmClassifier, testSet));

        for (int i=1;i<=100;i++){
            optimizer.iterate();
            System.out.println("after iteration "+i);
            System.out.println("train acc = "+ Accuracy.accuracy(bmmClassifier,dataSet));
            System.out.println("train overlap = "+ Overlap.overlap(bmmClassifier, dataSet));
            System.out.println("test acc = "+ Accuracy.accuracy(bmmClassifier,testSet));
            System.out.println("test overlap = "+ Overlap.overlap(bmmClassifier, testSet));
        }

        System.out.println("history = "+optimizer.getTerminator().getHistory());
        System.out.println(bmmClassifier);
    }

}