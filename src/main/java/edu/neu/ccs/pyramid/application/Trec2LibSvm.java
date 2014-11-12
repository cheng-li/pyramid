package edu.neu.ccs.pyramid.application;

import edu.neu.ccs.pyramid.configuration.Config;
import edu.neu.ccs.pyramid.dataset.ClfDataSet;
import edu.neu.ccs.pyramid.dataset.DataSetType;
import edu.neu.ccs.pyramid.dataset.LibSvmFormat;
import edu.neu.ccs.pyramid.dataset.TRECFormat;

import java.io.File;

/**
 * Created by chengli on 11/11/14.
 */
public class Trec2LibSvm {
    public static void main(String[] args) throws Exception {
        Config config = new Config(args[0]);
        String trecTrain = config.getString("trecTrain");
        String trecTest = config.getString("trecTest");
        String libSVMTrain = config.getString("libSVMTrain");
        String libSVMTest = config.getString("libSVMTest");


        ClfDataSet trainDataSet = TRECFormat.loadClfDataSet(new File(trecTrain),
                DataSetType.CLF_DENSE, true);

        ClfDataSet testDataset = TRECFormat.loadClfDataSet(new File(trecTest),
                DataSetType.CLF_DENSE, true);

        System.out.println("Translating Trec to libSVM for training...");
        LibSvmFormat.save(trainDataSet, libSVMTrain);
        System.out.println("Translating Trec to libSVM for testing...");
        LibSvmFormat.save(testDataset, libSVMTest);

    }
}
