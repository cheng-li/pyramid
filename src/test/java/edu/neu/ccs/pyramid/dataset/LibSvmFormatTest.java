package edu.neu.ccs.pyramid.dataset;

import edu.neu.ccs.pyramid.configuration.Config;

import java.io.File;

/**
 * Created by Rainicy on 10/28/14.
 */
public class LibSvmFormatTest {
    private static final Config config = new Config("configs/local.config");
    private static final String DATASETS = config.getString("input.datasets");
    private static final String TMP = config.getString("output.tmp");

    public static void main(String[] args) throws Exception {
        ClfDataSet dataSet = TRECFormat.loadClfDataSet(new File(DATASETS, "/spam/trec_data/train.trec"),
                DataSetType.CLF_DENSE, true);

        ClfDataSet testDataset = TRECFormat.loadClfDataSet(new File(DATASETS, "/spam/trec_data/test.trec"),
                DataSetType.CLF_DENSE, true);

        LibSvmFormat.save(dataSet, TMP + "spam_train.txt");
        LibSvmFormat.save(testDataset, TMP + "spam_test.txt");
    }
}
