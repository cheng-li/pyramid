package edu.neu.ccs.pyramid.dataset;

import edu.neu.ccs.pyramid.configuration.Config;
import edu.neu.ccs.pyramid.util.Serialization;
import junit.framework.TestCase;

import java.io.File;

/**
 * Created by chengli on 11/27/16.
 */
public class DenseDataSetTest {
    private static final Config config = new Config("config/local.properties");
    private static final String DATASETS = config.getString("input.datasets");
    private static final String TMP = config.getString("output.tmp");
    public static void main(String[] args) throws Exception{
        DenseDataSet dataSet = new DenseDataSet(10,5,false);
        dataSet.setFeatureValue(1,3,-0.9);
        dataSet.setFeatureValue(1,4,-0.9);
        dataSet.setFeatureValue(1,4,-60.9);
        dataSet.setFeatureValue(7,4,18);
        System.out.println(dataSet.getMetaInfo());
        System.out.println(dataSet);

        Serialization.serialize(dataSet, new File(TMP,"data.ser"));
        DenseDataSet loaded = (DenseDataSet)Serialization.deserialize(new File(TMP,"data.ser"));
        System.out.println(loaded);
        System.out.println(loaded.getMetaInfo());
    }
}