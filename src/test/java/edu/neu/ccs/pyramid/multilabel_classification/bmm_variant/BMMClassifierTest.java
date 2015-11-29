package edu.neu.ccs.pyramid.multilabel_classification.bmm_variant;

import edu.neu.ccs.pyramid.configuration.Config;
import edu.neu.ccs.pyramid.dataset.DataSetType;
import edu.neu.ccs.pyramid.dataset.MLClfDataSetBuilder;
import edu.neu.ccs.pyramid.dataset.MultiLabelClfDataSet;
import edu.neu.ccs.pyramid.dataset.TRECFormat;
import edu.neu.ccs.pyramid.eval.Accuracy;
import edu.neu.ccs.pyramid.eval.Overlap;
import edu.neu.ccs.pyramid.util.BernoulliDistribution;
import org.apache.commons.math3.distribution.NormalDistribution;


import java.io.File;

/**
 * Created by Rainicy on 10/24/15.
 */
public class BMMClassifierTest {

    private static final Config config = new Config("config/local.config");
    private static final String DATASETS = config.getString("input.datasets");
    private static final String TMP = config.getString("output.tmp");
    public static void main(String[] args) throws Exception{
        test1();
    }

    private static void test1() throws Exception{
        MultiLabelClfDataSet dataSet = TRECFormat.loadMultiLabelClfDataSet(new File(DATASETS, "/spam/trec_data/train.trec"),
                DataSetType.ML_CLF_SPARSE, true);
        MultiLabelClfDataSet testSet = TRECFormat.loadMultiLabelClfDataSet(new File(DATASETS, "/spam/trec_data/test.trec"),
                DataSetType.ML_CLF_SPARSE, true);
        BMMClassifier bmmClassifier = new BMMClassifier(dataSet.getNumClasses(),4,dataSet.getNumFeatures());
        bmmClassifier.setPredictMode("dynamic");
        BMMOptimizer optimizer = new BMMOptimizer(bmmClassifier,dataSet,1,1);
        bmmClassifier.setNumSample(100);
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

    private static void test2() throws Exception {

        MultiLabelClfDataSet dataSet = MLClfDataSetBuilder.getBuilder()
                .numFeatures(2).numClasses(4).numDataPoints(1000).build();



        BernoulliDistribution bernoulliDistribution = new BernoulliDistribution(0.5);
        for (int n=0; n<dataSet.getNumDataPoints(); n++) {
            for (int m=0; m<dataSet.getNumFeatures(); m++) {
                int bit = bernoulliDistribution.sample();
                int flip = bit;
                if (Math.random()<0.1) {
                    flip = 1 - bit;
                }
                dataSet.setFeatureValue(n,m,bit);
                if (m == 0) {
                    if (flip == 0) {
                        dataSet.addLabel(n,0);

                    } else {
                        dataSet.addLabel(n,1);
                    }
                } else {
                    if (flip == 0) {
                        dataSet.addLabel(n,2);
                    } else {
                        dataSet.addLabel(n,3);
                    }
                }
            }
        }

        MultiLabelClfDataSet testSet = MLClfDataSetBuilder.getBuilder()
                .numFeatures(2).numClasses(4).numDataPoints(100).build();

        for (int n=0; n<testSet.getNumDataPoints(); n++) {
            for (int m=0; m<testSet.getNumFeatures(); m++) {
                int bit = bernoulliDistribution.sample();
                testSet.setFeatureValue(n,m,bit);
                int flip = bit;
                if (Math.random()<0.1) {
                    flip = 1 - bit;
                }
                if (m == 0) {
                    if (flip == 0) {
                        testSet.addLabel(n,0);
                    } else {
                        testSet.addLabel(n,1);
                    }
                } else {
                    if (flip == 0) {
                        testSet.addLabel(n,2);
                    } else {
                        testSet.addLabel(n,3);
                    }
                }
            }
        }

        int numClusters = 50;
        BMMClassifier bmmClassifier = new BMMClassifier(dataSet,numClusters);
        BMMInitializer.initialize(bmmClassifier,dataSet,1.0,1.0);

        BMMOptimizer optimizer = new BMMOptimizer(bmmClassifier,dataSet,10000,10000);
        for (int i=0; i<3; i++) {
            optimizer.iterate();
            System.out.print("i: " + i + "\t");
            System.out.print("objective: " + optimizer.getTerminator().getLastValue() + "\t");
            System.out.print("trainAcc: " + Accuracy.accuracy(bmmClassifier,dataSet) + "\t");
            System.out.println("testAcc: " + Accuracy.accuracy(bmmClassifier,testSet));
        }
        System.out.println(bmmClassifier.toString());
    }
}
