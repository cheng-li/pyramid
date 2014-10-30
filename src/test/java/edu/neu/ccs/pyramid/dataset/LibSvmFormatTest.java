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
//        ClfDataSet dataSet = TRECFormat.loadClfDataSet(new File(DATASETS, "/classic3/train.trec"),
//                DataSetType.CLF_DENSE, true);
//
//        ClfDataSet testDataset = TRECFormat.loadClfDataSet(new File(DATASETS, "/classic3/test.trec"),
//                DataSetType.CLF_DENSE, true);
//
//        LibSvmFormat.save(dataSet, DATASETS + "classic3/libSvmFormat/classic3_train.txt");
//        LibSvmFormat.save(testDataset, DATASETS + "classic3/libSvmFormat/classic3_test.txt");


        ClfDataSet dataSet = TRECFormat.loadClfDataSet(new File(DATASETS, "/classic3/classic3_exp11/train.trec"),
                DataSetType.CLF_DENSE, true);

        ClfDataSet testDataset = TRECFormat.loadClfDataSet(new File(DATASETS, "/classic3/classic3_exp11/test.trec"),
                DataSetType.CLF_DENSE, true);

        LibSvmFormat.save(dataSet, DATASETS + "classic3/classic3_exp11/libSvmFormat/classic3_train.txt");
        LibSvmFormat.save(testDataset, DATASETS + "classic3/classic3_exp11/libSvmFormat/classic3_test.txt");

    }
}
