package edu.neu.ccs.pyramid.application;

import edu.neu.ccs.pyramid.configuration.Config;
import edu.neu.ccs.pyramid.dataset.ClfDataSet;
import edu.neu.ccs.pyramid.dataset.LibSvmFormat;
import edu.neu.ccs.pyramid.dataset.TRECFormat;

/**
 * Created by chengli on 5/30/15.
 */
public class LibSvm2Trec {
    public static void main(String[] args) throws Exception{
        Config config = new Config(args[0]);
        System.out.println(config);
        String libSvmFile = config.getString("libSvmFile");
        String trecFile = config.getString("trecFile");
        int numFeatures = config.getInt("numFeatures");
        int numClasses = config.getInt("numClasses");
        boolean dense = config.getBoolean("dense");
        ClfDataSet dataSet = LibSvmFormat.loadClfDataSet(libSvmFile,numFeatures,numClasses,dense);
        TRECFormat.save(dataSet,trecFile);
    }
}
