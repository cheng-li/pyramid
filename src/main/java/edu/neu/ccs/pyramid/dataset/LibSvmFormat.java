package edu.neu.ccs.pyramid.dataset;

import java.io.IOException;

/**
 * Created by chengli on 10/28/14.
 */
public class LibSvmFormat {

    public static void save(ClfDataSet dataSet, String libSvmFile){

    }

    public static void save(RegDataSet dataSet, String libSvmFile){

    }

    public static void save(MultiLabelClfDataSet dataSet, String libSvmFile){

    }

    public static ClfDataSet loadClfDataSet(String libSvmFile, DataSetType dataSetType,
                                            boolean loadSettings) throws IOException, ClassNotFoundException {
        return null;
    }

    public static RegDataSet loadRegDataSet(String libSvmFile, DataSetType dataSetType,
                                            boolean loadSettings) throws IOException, ClassNotFoundException {
        return null;
    }

    public static MultiLabelClfDataSet loaMultiLabelClfDataSet(String libSvmFile, DataSetType dataSetType,
                                                               boolean loadSettings) throws IOException, ClassNotFoundException {
        return null;
    }
}
