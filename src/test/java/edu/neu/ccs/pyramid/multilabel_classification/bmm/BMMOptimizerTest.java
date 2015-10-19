package edu.neu.ccs.pyramid.multilabel_classification.bmm;

import edu.neu.ccs.pyramid.configuration.Config;
import edu.neu.ccs.pyramid.dataset.*;
import edu.neu.ccs.pyramid.eval.Accuracy;
import edu.neu.ccs.pyramid.eval.Overlap;
import edu.neu.ccs.pyramid.util.BernoulliDistribution;

import java.io.File;


public class BMMOptimizerTest {
    private static final Config config = new Config("config/local.config");
    private static final String DATASETS = config.getString("input.datasets");
    private static final String TMP = config.getString("output.tmp");
    public static void main(String[] args) throws Exception{
        test4();
    }

    private static void test1() throws Exception{
        MultiLabelClfDataSet dataSet = TRECFormat.loadMultiLabelClfDataSet(new File(DATASETS, "spam/trec_data/train.trec"),
                DataSetType.ML_CLF_SPARSE, true);
        MultiLabelClfDataSet testSet = TRECFormat.loadMultiLabelClfDataSet(new File(DATASETS, "spam/trec_data/test.trec"),
                DataSetType.ML_CLF_SPARSE, true);
        int numClusters = 2;
        BMMClassifier bmmClassifier = new BMMClassifier(dataSet.getNumClasses(),numClusters,dataSet.getNumFeatures());
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


    private static void test3() throws Exception{
        int numCluster = 20;
        MultiLabelClfDataSet dataSet = TRECFormat.loadMultiLabelClfDataSet(new File(DATASETS, "20newsgroup/1/train.trec"),
                DataSetType.ML_CLF_SPARSE, true);
        MultiLabelClfDataSet testSet = TRECFormat.loadMultiLabelClfDataSet(new File(DATASETS, "20newsgroup/1/test.trec"),
                DataSetType.ML_CLF_SPARSE, true);
        BMMClassifier bmmClassifier = new BMMClassifier(dataSet.getNumClasses(),numCluster,dataSet.getNumFeatures());
        BMMOptimizer optimizer = new BMMOptimizer(bmmClassifier,dataSet,10000);
        bmmClassifier.setNumSample(100);

        System.out.println("after initialization");
        System.out.println("train acc = "+ Accuracy.accuracy(bmmClassifier,dataSet));
        System.out.println("test acc = "+ Accuracy.accuracy(bmmClassifier,testSet));

        for (int i=1;i<=10;i++){
            optimizer.iterate();
            System.out.println("after iteration "+i);
            System.out.println("objective = "+optimizer.getTerminator().getLastValue());
            System.out.println("train acc = "+ Accuracy.accuracy(bmmClassifier,dataSet));
            System.out.println("train overlap = "+ Overlap.overlap(bmmClassifier, dataSet));
            System.out.println("test acc = "+ Accuracy.accuracy(bmmClassifier,testSet));
            System.out.println("test overlap = "+ Overlap.overlap(bmmClassifier, testSet));
        }


        System.out.println("history = "+optimizer.getTerminator().getHistory());
        System.out.println(bmmClassifier);
    }

    private static void test4(){
        MultiLabelClfDataSet dataSet = MLClfDataSetBuilder.getBuilder()
                .numFeatures(10).numClasses(10).numDataPoints(1000)
                .build();
        BernoulliDistribution bernoulliDistribution = new BernoulliDistribution(0.5);
        for (int i=0;i<dataSet.getNumDataPoints();i++){
            for (int j=0;j<dataSet.getNumFeatures();j++){
                int bit = bernoulliDistribution.sample();

                if (bit==1){
                    dataSet.setFeatureValue(i,j,bit);
                    dataSet.addLabel(i,j);
                }
            }
        }

        int numClusters = 100;
        BMMClassifier bmmClassifier = new BMMClassifier(dataSet.getNumClasses(),numClusters,dataSet.getNumFeatures());
        bmmClassifier.setNumSample(100);
        BMMInitializer bmmInitializer = new BMMInitializer();
        bmmInitializer.initialize(bmmClassifier,dataSet);

        System.out.println("after initialization");
        System.out.println("train acc = "+ Accuracy.accuracy(bmmClassifier,dataSet));

        BMMOptimizer optimizer = new BMMOptimizer(bmmClassifier,dataSet,10000);
        for (int i=1;i<=10;i++){
            optimizer.iterate();
            System.out.println("after iteration "+i);
            System.out.println("objective = "+optimizer.getTerminator().getLastValue());
            System.out.println("train acc = "+ Accuracy.accuracy(bmmClassifier,dataSet));
        }
        System.out.println(bmmClassifier.toString());
        for (int k=0;k<numClusters;k++){
            System.out.println("cluster "+k);
            System.out.println(bmmClassifier.logisticRegression.getWeights().getWeightsWithoutBiasForClass(k));
        }

    }

}