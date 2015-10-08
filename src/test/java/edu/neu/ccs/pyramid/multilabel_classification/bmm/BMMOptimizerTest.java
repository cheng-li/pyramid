package edu.neu.ccs.pyramid.multilabel_classification.bmm;

import edu.neu.ccs.pyramid.configuration.Config;
import edu.neu.ccs.pyramid.dataset.ClfDataSet;
import edu.neu.ccs.pyramid.dataset.DataSetType;
import edu.neu.ccs.pyramid.dataset.MultiLabelClfDataSet;
import edu.neu.ccs.pyramid.dataset.TRECFormat;
import edu.neu.ccs.pyramid.eval.Accuracy;

import java.io.File;

import static org.junit.Assert.*;

public class BMMOptimizerTest {
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
        BMMClassifier bmmClassifier = new BMMClassifier(dataSet.getNumClasses(),2,dataSet.getNumFeatures());
        BMMOptimizer optimizer = new BMMOptimizer(bmmClassifier,dataSet,10000);

        System.out.println("after initialization");
        System.out.println("train acc = "+ Accuracy.accuracy(bmmClassifier,dataSet));
        System.out.println("test acc = "+ Accuracy.accuracy(bmmClassifier,testSet));

        for (int i=0;i<100;i++){
            optimizer.iterate();
        }

        System.out.println("after training");
        System.out.println("train acc = "+ Accuracy.accuracy(bmmClassifier,dataSet));
        System.out.println("test acc = "+ Accuracy.accuracy(bmmClassifier,testSet));
    }

}