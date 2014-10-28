package edu.neu.ccs.pyramid.dataset;

import java.io.IOException;

/**
 * Created by chengli on 10/28/14.
 */
public class LibSvmFormat {

    public static void save(ClfDataSet dataSet, String trecFile){

    }

    public static void save(RegDataSet dataSet, String trecFile){

    }

    public static void save(MultiLabelClfDataSet dataSet, String trecFile){

    }

    public static ClfDataSet loadClfDataSet(String trecFile, DataSetType dataSetType,
                                            boolean loadSettings) throws IOException, ClassNotFoundException {
        return null;
    }

    public static RegDataSet loadRegDataSet(String trecFile, DataSetType dataSetType,
                                            boolean loadSettings) throws IOException, ClassNotFoundException {
        return null;
    }

    public static MultiLabelClfDataSet loaMultiLabelClfDataSet(String trecFile, DataSetType dataSetType,
                                                               boolean loadSettings) throws IOException, ClassNotFoundException {
        return null;
    }
}
