package edu.neu.ccs.pyramid.dataset;

import edu.neu.ccs.pyramid.configuration.Config;

import java.io.File;

public class ArffFormatTest {
    private static final Config config = new Config("config/local.config");
    private static final String DATASETS = config.getString("input.datasets");
    private static final String TMP = config.getString("output.tmp");

    public static void main(String[] args) throws Exception {
        test_train();
        test_test();
    }

    private static void test_train() throws Exception{

        ClfDataSet dataSet = TRECFormat.loadClfDataSet(new File(DATASETS,"/classic3/classic3_exp11/train.trec"),
                DataSetType.CLF_DENSE, true);
        ArffFormat.save(dataSet, new File(TMP,"/classic3/classic3_exp11/ArffFormat/train"));
    }

    private static void test_test() throws Exception{

        ClfDataSet dataSet = TRECFormat.loadClfDataSet(new File(DATASETS,"/classic3/classic3_exp11/test.trec"),
        DataSetType.CLF_DENSE, true);
        ArffFormat.save(dataSet, new File(TMP,"/classic3/classic3_exp11/ArffFormat/test"));
    }
}