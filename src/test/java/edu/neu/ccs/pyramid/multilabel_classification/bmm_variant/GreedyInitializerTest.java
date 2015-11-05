package edu.neu.ccs.pyramid.multilabel_classification.bmm_variant;

import edu.neu.ccs.pyramid.configuration.Config;
import edu.neu.ccs.pyramid.dataset.DataSetType;
import edu.neu.ccs.pyramid.dataset.MultiLabelClfDataSet;
import edu.neu.ccs.pyramid.dataset.TRECFormat;
import edu.neu.ccs.pyramid.eval.Accuracy;
import edu.neu.ccs.pyramid.eval.Overlap;

import java.io.File;

import static org.junit.Assert.*;

public class GreedyInitializerTest {
    private static final Config config = new Config("config/local.config");
    private static final String DATASETS = config.getString("input.datasets");
    private static final String TMP = config.getString("output.tmp");
    public static void main(String[] args) throws Exception{
        test2();
    }

    private static void test1() throws Exception{
        MultiLabelClfDataSet dataSet = TRECFormat.loadMultiLabelClfDataSet(new File(DATASETS, "/spam/trec_data/train.trec"),
                DataSetType.ML_CLF_SPARSE, true);
        MultiLabelClfDataSet testSet = TRECFormat.loadMultiLabelClfDataSet(new File(DATASETS, "/spam/trec_data/test.trec"),
                DataSetType.ML_CLF_SPARSE, true);
        int numClusters = 10;
        double variance = 1000;
        GreedyInitializer greedyInitializer = new GreedyInitializer(dataSet,numClusters,variance);
        greedyInitializer.train();
        System.out.println(greedyInitializer);
    }


    private static void test2() throws Exception{
        MultiLabelClfDataSet dataSet = TRECFormat.loadMultiLabelClfDataSet(new File(DATASETS, "ohsumed/3/train.trec"), DataSetType.ML_CLF_SPARSE, true);
        int numClusters = 10;
        double variance = 1000;
        GreedyInitializer greedyInitializer = new GreedyInitializer(dataSet,numClusters,variance);
        greedyInitializer.train();
        System.out.println(greedyInitializer);
    }
}