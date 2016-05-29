package edu.neu.ccs.pyramid.multilabel_classification.cbm;

import edu.neu.ccs.pyramid.configuration.Config;
import edu.neu.ccs.pyramid.dataset.DataSetType;
import edu.neu.ccs.pyramid.dataset.MLClfDataSetBuilder;
import edu.neu.ccs.pyramid.dataset.MultiLabelClfDataSet;
import edu.neu.ccs.pyramid.dataset.TRECFormat;
import edu.neu.ccs.pyramid.eval.Accuracy;
import edu.neu.ccs.pyramid.eval.Overlap;
import edu.neu.ccs.pyramid.util.BernoulliDistribution;


import java.io.File;

/**
 * Created by Rainicy on 10/24/15.
 */
public class CBMTest {

    private static final Config config = new Config("config/local.properties");
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

        int numClusters = 4;
        CBM cbm = CBM.getBuilder()
                .setNumClasses(dataSet.getNumClasses())
                .setNumFeatures(dataSet.getNumFeatures())
                .setNumClusters(numClusters)
                .setMultiClassClassifierType("lr")
                .setBinaryClassifierType("boost")
                .build();

        cbm.setPredictMode("dynamic");
        CBMOptimizer optimizer = new CBMOptimizer(cbm,dataSet);
        optimizer.setPriorVarianceBinary(10);
        optimizer.setPriorVarianceMultiClass(10);
        CBMInitializer.initialize(cbm,dataSet,optimizer);
        cbm.setNumSample(100);
        System.out.println("num cluster: " + cbm.numClusters);

        System.out.println("after initialization");
        System.out.println("train acc = "+ Accuracy.accuracy(cbm, dataSet));
        System.out.println("test acc = "+ Accuracy.accuracy(cbm,testSet));


        for (int i=1;i<=5;i++){
            optimizer.iterate();
            System.out.print("iter : "+i + "\t");
            System.out.print("objective: "+optimizer.getTerminator().getLastValue() + "\t");
            System.out.print("trainAcc : "+ Accuracy.accuracy(cbm,dataSet)+ "\t");
            System.out.print("trainOver: "+ Overlap.overlap(cbm, dataSet)+ "\t");
            System.out.print("testAcc  : "+ Accuracy.accuracy(cbm,testSet)+ "\t");
            System.out.println("testOver : "+ Overlap.overlap(cbm, testSet)+ "\t");
        }


        System.out.println("history = "+optimizer.getTerminator().getHistory());
        System.out.println(cbm);
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

        int numClusters = 4;
        CBM cbm = CBM.getBuilder()
                .setNumClasses(dataSet.getNumClasses())
                .setNumFeatures(dataSet.getNumFeatures())
                .setNumClusters(numClusters)
                .setBinaryClassifierType("boost")
                .setMultiClassClassifierType("boost")
                .build();

        cbm.setPredictMode("dynamic");
        CBMOptimizer optimizer = new CBMOptimizer(cbm,dataSet);
        optimizer.setPriorVarianceBinary(10);
        optimizer.setPriorVarianceMultiClass(10);
        CBMInitializer.initialize(cbm,dataSet,optimizer);
        for (int i=0; i<3; i++) {
            optimizer.iterate();
            System.out.print("i: " + i + "\t");
            System.out.print("objective: " + optimizer.getTerminator().getLastValue() + "\t");
            System.out.print("trainAcc: " + Accuracy.accuracy(cbm,dataSet) + "\t");
            System.out.println("testAcc: " + Accuracy.accuracy(cbm,testSet));
        }
        System.out.println(cbm.toString());
    }


    private static void test3() throws Exception{
        MultiLabelClfDataSet dataSet = TRECFormat.loadMultiLabelClfDataSet(new File(DATASETS, "ohsumed/3/train.trec"),
                DataSetType.ML_CLF_SPARSE, true);
        MultiLabelClfDataSet testSet = TRECFormat.loadMultiLabelClfDataSet(new File(DATASETS, "ohsumed/3/test.trec"),
                DataSetType.ML_CLF_SPARSE, true);
        int numClusters = 4;
        CBM cbm = CBM.getBuilder()
                .setNumClasses(dataSet.getNumClasses())
                .setNumFeatures(dataSet.getNumFeatures())
                .setNumClusters(numClusters)
                .setBinaryClassifierType("lr")
                .setMultiClassClassifierType("boost")
                .build();

        cbm.setPredictMode("dynamic");
        CBMOptimizer optimizer = new CBMOptimizer(cbm,dataSet);
        optimizer.setPriorVarianceBinary(10);
        optimizer.setPriorVarianceMultiClass(10);
        CBMInitializer.initialize(cbm,dataSet,optimizer);

        cbm.setNumSample(100);
        System.out.println("num cluster: " + cbm.numClusters);

        System.out.println("after initialization");
        System.out.println("train acc = "+ Accuracy.accuracy(cbm, dataSet));
        System.out.println("test acc = "+ Accuracy.accuracy(cbm,testSet));


        for (int i=1;i<=30;i++){
            optimizer.iterate();
            System.out.print("iter : "+i + "\t");
            System.out.print("objective: "+optimizer.getTerminator().getLastValue() + "\t");
            System.out.print("trainAcc : "+ Accuracy.accuracy(cbm,dataSet)+ "\t");
            System.out.print("trainOver: "+ Overlap.overlap(cbm, dataSet)+ "\t");
            System.out.print("testAcc  : "+ Accuracy.accuracy(cbm,testSet)+ "\t");
            System.out.println("testOver : "+ Overlap.overlap(cbm, testSet)+ "\t");
        }


        System.out.println("history = "+optimizer.getTerminator().getHistory());
        System.out.println(cbm);
    }

    private static void test4() throws Exception{
        MultiLabelClfDataSet dataSet = TRECFormat.loadMultiLabelClfDataSet(new File(DATASETS, "scene/train"),
                DataSetType.ML_CLF_SPARSE, true);
        MultiLabelClfDataSet testSet = TRECFormat.loadMultiLabelClfDataSet(new File(DATASETS, "scene/test"),
                DataSetType.ML_CLF_SPARSE, true);
        int numClusters = 4;
        CBM cbm = CBM.getBuilder()
                .setNumClasses(dataSet.getNumClasses())
                .setNumFeatures(dataSet.getNumFeatures())
                .setNumClusters(numClusters)
                .setBinaryClassifierType("lr")
                .setMultiClassClassifierType("boost")
                .build();

        cbm.setPredictMode("dynamic");
        CBMOptimizer optimizer = new CBMOptimizer(cbm,dataSet);
        optimizer.setPriorVarianceBinary(10);
        optimizer.setPriorVarianceMultiClass(10);
        CBMInitializer.initialize(cbm,dataSet,optimizer);
        cbm.setNumSample(100);
        System.out.println("num cluster: " + cbm.numClusters);

        System.out.println("after initialization");
        System.out.println("train acc = "+ Accuracy.accuracy(cbm, dataSet));
        System.out.println("test acc = "+ Accuracy.accuracy(cbm,testSet));


        for (int i=1;i<=30;i++){
            optimizer.iterate();
            System.out.print("iter : "+i + "\t");
            System.out.print("objective: "+optimizer.getTerminator().getLastValue() + "\t");
            System.out.print("trainAcc : "+ Accuracy.accuracy(cbm,dataSet)+ "\t");
            System.out.print("trainOver: "+ Overlap.overlap(cbm, dataSet)+ "\t");
            System.out.print("testAcc  : "+ Accuracy.accuracy(cbm,testSet)+ "\t");
            System.out.println("testOver : "+ Overlap.overlap(cbm, testSet)+ "\t");
        }


        System.out.println("history = "+optimizer.getTerminator().getHistory());
        System.out.println(cbm);
    }
}
