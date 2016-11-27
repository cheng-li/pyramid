package edu.neu.ccs.pyramid.dataset;

import edu.neu.ccs.pyramid.configuration.Config;
import edu.neu.ccs.pyramid.util.Serialization;

import java.io.File;

public class DenseMLClfDataSetTest {
    private static final Config config = new Config("config/local.properties");
    private static final String DATASETS = config.getString("input.datasets");
    private static final String TMP = config.getString("output.tmp");

    public static void main(String[] args) throws Exception{
        test3();
    }

    static void test1(){
        MultiLabelClfDataSet dataSet = new DenseMLClfDataSet(10,5,false,3);
        dataSet.setFeatureValue(1,3,-0.9);
        dataSet.setFeatureValue(1,4,-0.9);
        dataSet.setFeatureValue(1,4,-60.9);
        dataSet.setFeatureValue(7,4,18);
        dataSet.addLabel(0,1);
        dataSet.addLabel(0,2);
        dataSet.addLabel(1,0);
        System.out.println(dataSet.getMetaInfo());
        System.out.println(dataSet);
    }


    static void test2() throws Exception{
        MultiLabelClfDataSet dataSet = new DenseMLClfDataSet(10,5,false,3);
        dataSet.setFeatureValue(1,3,-0.9);
        dataSet.setFeatureValue(1,4,-0.9);
        dataSet.setFeatureValue(1,4,-60.9);
        dataSet.setFeatureValue(7,4,18);
        dataSet.addLabel(0,1);
        dataSet.addLabel(0,2);
        dataSet.addLabel(1,0);
        System.out.println(dataSet.getMetaInfo());
        System.out.println(dataSet);

        Serialization.serialize(dataSet, new File(TMP,"data.ser"));
        MultiLabelClfDataSet loaded = (MultiLabelClfDataSet)Serialization.deserialize(new File(TMP,"data.ser"));
        System.out.println(loaded);
        System.out.println(loaded.getMetaInfo());

    }


    private static void test3() throws Exception{
        MultiLabelClfDataSet dataSet = TRECFormat.loadMultiLabelClfDataSet(new File(DATASETS, "medical/train"),
                DataSetType.ML_CLF_DENSE, true);

        Serialization.serialize(dataSet, new File(TMP,"data.ser"));
        MultiLabelClfDataSet loaded = (MultiLabelClfDataSet)Serialization.deserialize(new File(TMP,"data.ser"));

        System.out.println(dataSet.toString().equals(loaded.toString()));

    }



}