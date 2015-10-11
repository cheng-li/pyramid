package edu.neu.ccs.pyramid.multilabel_classification.bmm;

import edu.neu.ccs.pyramid.configuration.Config;
import edu.neu.ccs.pyramid.dataset.DataSetType;
import edu.neu.ccs.pyramid.dataset.MultiLabelClfDataSet;
import edu.neu.ccs.pyramid.dataset.TRECFormat;
import edu.neu.ccs.pyramid.eval.Accuracy;
import edu.neu.ccs.pyramid.eval.Overlap;

import java.io.File;

import static org.junit.Assert.*;

public class BMMInitializerTest {
    private static final Config config = new Config("config/local.config");
    private static final String DATASETS = config.getString("input.datasets");
    private static final String TMP = config.getString("output.tmp");
    public static void main(String[] args) throws Exception{
        test1();
    }

    private static void test1() throws Exception{

        MultiLabelClfDataSet trainSet = TRECFormat.loadMultiLabelClfDataSet(new File(DATASETS,"reuters/train"),
                DataSetType.ML_CLF_SPARSE, true);

        MultiLabelClfDataSet testSet = TRECFormat.loadMultiLabelClfDataSet(new File(DATASETS,"reuters/test"),
                DataSetType.ML_CLF_SPARSE, true);

        int numClusters = 50;
        int numSamples = 100;

        BMMClassifier bmmClassifier = new BMMClassifier(trainSet.getNumClasses(),numClusters,trainSet.getNumFeatures());
        bmmClassifier.setNumSample(numSamples);
        System.out.print("random init" + "\t" );
        System.out.print("trainAcc : "+ Accuracy.accuracy(bmmClassifier, trainSet) + "\t");
        System.out.print("trainOver: "+ Overlap.overlap(bmmClassifier, trainSet) + "\t");
        System.out.print("testACC  : "+ Accuracy.accuracy(bmmClassifier,testSet) + "\t");
        System.out.println("testOver : "+ Overlap.overlap(bmmClassifier, testSet) + "\t");

        BMMInitializer bmmInitializer = new BMMInitializer();
        bmmInitializer.initialize(bmmClassifier,trainSet);
        System.out.print("pure-label" + "\t");
        System.out.print("trainAcc : "+ Accuracy.accuracy(bmmClassifier,trainSet)+ "\t");
        System.out.print("trainOver: "+ Overlap.overlap(bmmClassifier, trainSet)+ "\t");
        System.out.print("testAcc  : "+ Accuracy.accuracy(bmmClassifier,testSet)+ "\t");
        System.out.println("testOver : "+ Overlap.overlap(bmmClassifier, testSet)+ "\t");
    }

}