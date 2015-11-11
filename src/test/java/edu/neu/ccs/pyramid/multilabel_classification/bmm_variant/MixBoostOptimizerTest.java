package edu.neu.ccs.pyramid.multilabel_classification.bmm_variant;

import edu.neu.ccs.pyramid.configuration.Config;
import edu.neu.ccs.pyramid.dataset.DataSetType;
import edu.neu.ccs.pyramid.dataset.MultiLabelClfDataSet;
import edu.neu.ccs.pyramid.dataset.TRECFormat;
import edu.neu.ccs.pyramid.eval.Accuracy;
import edu.neu.ccs.pyramid.eval.Overlap;

import java.io.File;

import static org.junit.Assert.*;

public class MixBoostOptimizerTest {
    private static final Config config = new Config("config/local.config");
    private static final String DATASETS = config.getString("input.datasets");
    private static final String TMP = config.getString("output.tmp");
    public static void main(String[] args) throws Exception{
        test3();
    }

    private static void test1() throws Exception{
        MultiLabelClfDataSet dataSet = TRECFormat.loadMultiLabelClfDataSet(new File(DATASETS, "/spam/trec_data/train.trec"),
                DataSetType.ML_CLF_SPARSE, true);
        MultiLabelClfDataSet testSet = TRECFormat.loadMultiLabelClfDataSet(new File(DATASETS, "/spam/trec_data/test.trec"),
                DataSetType.ML_CLF_SPARSE, true);

        int numClusters = 1;
        BMMClassifier bmmClassifier = BMMClassifier.newMixBoost(dataSet.getNumClasses(), numClusters, dataSet.getNumFeatures());
        MixBoostOptimizer optimizer = new MixBoostOptimizer(bmmClassifier,dataSet);
        System.out.println("num cluster: " + bmmClassifier.numClusters);

        System.out.println("after initialization");
        System.out.println("train acc = "+ Accuracy.accuracy(bmmClassifier, dataSet));
        System.out.println("test acc = "+ Accuracy.accuracy(bmmClassifier,testSet));


        for (int i=1;i<=30;i++){
            optimizer.iterate();
            System.out.print("iter : "+i + "\t");
            System.out.print("objective: "+optimizer.getTerminator().getLastValue() + "\t");
            System.out.print("trainAcc : "+ Accuracy.accuracy(bmmClassifier,dataSet)+ "\t");
            System.out.print("trainOver: "+ Overlap.overlap(bmmClassifier, dataSet)+ "\t");
            System.out.print("testAcc  : "+ Accuracy.accuracy(bmmClassifier,testSet)+ "\t");
            System.out.println("testOver : "+ Overlap.overlap(bmmClassifier, testSet)+ "\t");
        }


        System.out.println("history = "+optimizer.getTerminator().getHistory());
        System.out.println(bmmClassifier);
    }

    private static void test2() throws Exception{
        MultiLabelClfDataSet dataSet = TRECFormat.loadMultiLabelClfDataSet(new File(DATASETS, "/spam/trec_data/train.trec"),
                DataSetType.ML_CLF_SPARSE, true);
        MultiLabelClfDataSet testSet = TRECFormat.loadMultiLabelClfDataSet(new File(DATASETS, "/spam/trec_data/test.trec"),
                DataSetType.ML_CLF_SPARSE, true);

        int numClusters = 2;
        BMMClassifier bmmClassifier = BMMClassifier.newMixBoost(dataSet.getNumClasses(),numClusters,dataSet.getNumFeatures());
        MixBoostOptimizer optimizer = new MixBoostOptimizer(bmmClassifier,dataSet);
        System.out.println("num cluster: " + bmmClassifier.numClusters);

        System.out.println("after random initialization");
        System.out.println("train acc = "+ Accuracy.accuracy(bmmClassifier, dataSet));
        System.out.println("test acc = "+ Accuracy.accuracy(bmmClassifier,testSet));

        MixBoostInitializer.initialize(bmmClassifier,dataSet);
        System.out.println("after initializer initialization");
        System.out.println("train acc = "+ Accuracy.accuracy(bmmClassifier, dataSet));
        System.out.println("test acc = "+ Accuracy.accuracy(bmmClassifier,testSet));


        for (int i=1;i<=30;i++){
            optimizer.iterate();
            System.out.print("iter : "+i + "\t");
            System.out.print("objective: "+optimizer.getTerminator().getLastValue() + "\t");
            System.out.print("trainAcc : "+ Accuracy.accuracy(bmmClassifier,dataSet)+ "\t");
            System.out.print("trainOver: "+ Overlap.overlap(bmmClassifier, dataSet)+ "\t");
            System.out.print("testAcc  : "+ Accuracy.accuracy(bmmClassifier,testSet)+ "\t");
            System.out.println("testOver : "+ Overlap.overlap(bmmClassifier, testSet)+ "\t");
        }


        System.out.println("history = "+optimizer.getTerminator().getHistory());
        System.out.println(bmmClassifier);
    }

    private static void test3() throws Exception{
        MultiLabelClfDataSet dataSet = TRECFormat.loadMultiLabelClfDataSet(new File(DATASETS, "/medical/train"),
                DataSetType.ML_CLF_SPARSE, true);
        MultiLabelClfDataSet testSet = TRECFormat.loadMultiLabelClfDataSet(new File(DATASETS, "/medical/test"),
                DataSetType.ML_CLF_SPARSE, true);

        int numClusters = 2;
        BMMClassifier bmmClassifier = BMMClassifier.newMixBoost(dataSet.getNumClasses(),numClusters,dataSet.getNumFeatures());
        MixBoostOptimizer optimizer = new MixBoostOptimizer(bmmClassifier,dataSet);
        System.out.println("num cluster: " + bmmClassifier.numClusters);

        System.out.println("after random initialization");
        System.out.println("train acc = "+ Accuracy.accuracy(bmmClassifier, dataSet));
        System.out.println("test acc = "+ Accuracy.accuracy(bmmClassifier,testSet));

        MixBoostInitializer.initialize(bmmClassifier,dataSet);
        System.out.println("after initializer initialization");
        System.out.println("train acc = "+ Accuracy.accuracy(bmmClassifier, dataSet));
        System.out.println("test acc = "+ Accuracy.accuracy(bmmClassifier,testSet));


        for (int i=1;i<=30;i++){
            optimizer.iterate();
            System.out.print("iter : "+i + "\t");
            System.out.print("objective: "+optimizer.getTerminator().getLastValue() + "\t");
            System.out.print("trainAcc : "+ Accuracy.accuracy(bmmClassifier,dataSet)+ "\t");
            System.out.print("trainOver: "+ Overlap.overlap(bmmClassifier, dataSet)+ "\t");
            System.out.print("testAcc  : "+ Accuracy.accuracy(bmmClassifier,testSet)+ "\t");
            System.out.println("testOver : "+ Overlap.overlap(bmmClassifier, testSet)+ "\t");
        }


        System.out.println("history = "+optimizer.getTerminator().getHistory());
        System.out.println(bmmClassifier);
    }

}