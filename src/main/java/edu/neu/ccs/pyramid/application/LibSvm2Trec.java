package edu.neu.ccs.pyramid.application;

import edu.neu.ccs.pyramid.configuration.Config;
import edu.neu.ccs.pyramid.dataset.*;

import java.io.IOException;
import java.util.List;

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
            case "mlclf":
                translateMLClfData(config);
        }
    }

    private static void translateMLClfData(Config config) throws IOException, ClassNotFoundException {
        List<String> libSvmFiles = config.getStrings("libsvm");
        List<String> trecFiles = config.getStrings("trec");
        boolean dense = config.getBoolean("dense");
        int numFeatures = config.getInt("numFeatures");
        int numClasses = config.getInt("numClasses");
        for (int i=0; i<libSvmFiles.size(); i++) {
            String libSvmFile = libSvmFiles.get(i);
            String trecFile = trecFiles.get(i);
            System.out.println("translating: " + libSvmFile);
            MultiLabelClfDataSet dataSet = LibSvmFormat.loadMultiLabelClfDataSet(libSvmFile,dense,numFeatures,numClasses);
            TRECFormat.save(dataSet, trecFile);
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
