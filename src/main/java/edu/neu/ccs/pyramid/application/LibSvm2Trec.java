package edu.neu.ccs.pyramid.application;

import edu.neu.ccs.pyramid.configuration.Config;
import edu.neu.ccs.pyramid.dataset.ClfDataSet;
import edu.neu.ccs.pyramid.dataset.LibSvmFormat;
import edu.neu.ccs.pyramid.dataset.RegDataSet;
import edu.neu.ccs.pyramid.dataset.TRECFormat;

/**
 * Created by chengli on 5/30/15.
 */
public class LibSvm2Trec {
    public static void main(String[] args) throws Exception{
        Config config = new Config(args[0]);
        System.out.println(config);
        String dataType = config.getString("dataSetType");
        switch (dataType) {
            case "clf":
                translateClfData(config);
                break;
            case "reg":
                translateRegData(config);
                break;
        }

    }

    private static void translateClfData(Config config) throws Exception{
        String libSvmFile = config.getString("libSvmFile");
        String trecFile = config.getString("trecFile");
        int numFeatures = config.getInt("numFeatures");
        int numClasses = config.getInt("numClasses");
        boolean dense = config.getBoolean("dense");
        ClfDataSet dataSet = LibSvmFormat.loadClfDataSet(libSvmFile,numFeatures,numClasses,dense);
        TRECFormat.save(dataSet,trecFile);
    }

    private static void translateRegData(Config config) throws Exception{
        String libSvmFile = config.getString("libSvmFile");
        String trecFile = config.getString("trecFile");
        int numFeatures = config.getInt("numFeatures");
        boolean dense = config.getBoolean("dense");
        RegDataSet dataSet = LibSvmFormat.loadRegDataSet(libSvmFile, numFeatures, dense);
        TRECFormat.save(dataSet,trecFile);
    }
}
