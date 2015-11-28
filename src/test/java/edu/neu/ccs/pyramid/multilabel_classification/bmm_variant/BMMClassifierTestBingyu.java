package edu.neu.ccs.pyramid.multilabel_classification.bmm_variant;

import edu.neu.ccs.pyramid.configuration.Config;
import edu.neu.ccs.pyramid.dataset.DataSetType;
import edu.neu.ccs.pyramid.dataset.MultiLabelClfDataSet;
import edu.neu.ccs.pyramid.dataset.TRECFormat;
import edu.neu.ccs.pyramid.eval.Accuracy;
import edu.neu.ccs.pyramid.eval.Overlap;
import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.Vector;


import java.io.File;

/**
 * Created by Rainicy on 10/24/15.
 */
public class BMMClassifierTestBingyu {
    private static final Config config = new Config("/Users/Rainicy/Datasets/2.config");
    private static final String DATASETS = config.getString("input.datasets");
    public static void main(String[] args) throws Exception{
        test3();
    }

    private static void test3() throws Exception{
        double[] arr = {1, 2, 3};
        DenseVector vector = new DenseVector(arr);

        Vector newV = new DenseVector(vector,false);

        newV.set(2, 10);

        System.out.println("V1: " + vector);
        System.out.println("V2: " + newV);
    }
//
//    private static void test1() throws Exception{
//        MultiLabelClfDataSet dataSet = TRECFormat.loadMultiLabelClfDataSet(new File(DATASETS, "data_sets/train.trec"),
//                DataSetType.ML_CLF_SPARSE, true);
//        MultiLabelClfDataSet testSet = TRECFormat.loadMultiLabelClfDataSet(new File(DATASETS, "data_sets/test.trec"),
//                DataSetType.ML_CLF_SPARSE, true);
//        BMMClassifier bmmClassifier = new BMMClassifier(dataSet.getNumClasses(),1,dataSet.getNumFeatures());
//        BMMOptimizer optimizer = new BMMOptimizer(bmmClassifier,dataSet,1,1);
//        bmmClassifier.setNumSample(100);
//        System.out.println("num cluster: " + bmmClassifier.numClusters);
//
//        System.out.println("after initialization");
//        System.out.println("train acc = "+ Accuracy.accuracy(bmmClassifier, dataSet));
//        System.out.println("test acc = "+ Accuracy.accuracy(bmmClassifier,testSet));
//
//        for (int i=1;i<=10;i++){
//            optimizer.iterate();
//            System.out.print("iter : "+i + "\t");
//            System.out.print("objective: "+optimizer.getTerminator().getLastValue() + "\t");
//            System.out.print("trainAcc : "+ Accuracy.accuracy(bmmClassifier,dataSet)+ "\t");
//            System.out.print("trainOver: "+ Overlap.overlap(bmmClassifier, dataSet)+ "\t");
//            System.out.print("testAcc  : "+ Accuracy.accuracy(bmmClassifier,testSet)+ "\t");
//            System.out.println("testOver : "+ Overlap.overlap(bmmClassifier, testSet)+ "\t");
//        }
//
//
//        System.out.println("history = "+optimizer.getTerminator().getHistory());
//        System.out.println(bmmClassifier);
//    }

}
