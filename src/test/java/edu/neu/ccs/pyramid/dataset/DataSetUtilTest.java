package edu.neu.ccs.pyramid.dataset;

import edu.neu.ccs.pyramid.configuration.Config;

import java.io.File;

public class DataSetUtilTest {

    private static final Config config = new Config("configs/local.config");
    private static final String DATASETS = config.getString("input.datasets");
    private static final String TMP = config.getString("output.tmp");

    public static void main(String[] args) throws Exception{

        test6();
    }

    static void test1(){
        ClfDataSet clfDataSet = new SparseClfDataSet(5,3,false,6);
        clfDataSet.setFeatureValue(0,0,3.5);
        clfDataSet.setFeatureValue(1,2,5.5);
        clfDataSet.setFeatureValue(4,1,2.5);
        clfDataSet.setFeatureValue(4,2,5.5);
        clfDataSet.setLabel(0, 1);
        clfDataSet.setLabel(1,2);
        clfDataSet.setLabel(2,3);
        clfDataSet.setLabel(3,5);
        clfDataSet.setLabel(4,2);
        clfDataSet.getRow(0).putSetting(new DataSetting().setExtId("zero").setExtLabel("spam"));
        clfDataSet.getRow(1).putSetting(new DataSetting().setExtId("first").setExtLabel("non-spam"));
        clfDataSet.getRow(2).putSetting(new DataSetting().setExtId("second").setExtLabel("good"));
        clfDataSet.getRow(3).putSetting(new DataSetting().setExtId("third").setExtLabel("bad"));
        clfDataSet.getRow(4).putSetting(new DataSetting().setExtId("fourth").setExtLabel("iii"));
        clfDataSet.getColumn(0).putSetting(new FeatureSetting().setFeatureName("color").setFeatureType(FeatureType.BINARY));
        clfDataSet.getColumn(1).putSetting(new FeatureSetting().setFeatureName("age").setFeatureType(FeatureType.NUMERICAL));
        clfDataSet.getColumn(2).putSetting(new FeatureSetting().setFeatureName("income").setFeatureType(FeatureType.NUMERICAL));
        System.out.println(clfDataSet);
        ClfDataSet trimmed = DataSetUtil.trim(clfDataSet,2);
        System.out.println(trimmed);
    }

    static void test2(){
        ClfDataSet clfDataSet = new SparseClfDataSet(5,3,false,3);
        clfDataSet.setFeatureValue(0,0,3.5);
        clfDataSet.setFeatureValue(1,2,5.5);
        clfDataSet.setFeatureValue(4,1,2.5);
        clfDataSet.setFeatureValue(4,2,5.5);
        clfDataSet.setLabel(0,1);
        clfDataSet.setLabel(1,0);
        clfDataSet.setLabel(2,1);
        clfDataSet.setLabel(3,2);
        clfDataSet.setLabel(4,2);
        clfDataSet.getRow(0).putSetting(new DataSetting().setExtId("zero").setExtLabel("spam"));
        clfDataSet.getRow(1).putSetting(new DataSetting().setExtId("first").setExtLabel("non-spam"));
        clfDataSet.getRow(2).putSetting(new DataSetting().setExtId("second").setExtLabel("good"));
        clfDataSet.getRow(3).putSetting(new DataSetting().setExtId("third").setExtLabel("bad"));
        clfDataSet.getRow(4).putSetting(new DataSetting().setExtId("fourth").setExtLabel("iii"));
        clfDataSet.getColumn(0).putSetting(new FeatureSetting().setFeatureName("color").setFeatureType(FeatureType.BINARY));
        clfDataSet.getColumn(1).putSetting(new FeatureSetting().setFeatureName("age").setFeatureType(FeatureType.NUMERICAL));
        clfDataSet.getColumn(2).putSetting(new FeatureSetting().setFeatureName("income").setFeatureType(FeatureType.NUMERICAL));
        System.out.println(clfDataSet);
        System.out.println("bootstrapped sample");
        System.out.println(DataSetUtil.bootstrap(clfDataSet));

    }

    static void test3() throws Exception{
        ClfDataSet dataSet = TRECFormat.loadClfDataSet(new File(DATASETS,"/spam/trec_data/train.trec"),
                DataSetType.CLF_DENSE,true);
        DataSetUtil.dumpDataSettings(dataSet,new File(TMP,"data_settings.txt"));
        DataSetUtil.dumpFeatureSettings(dataSet,new File(TMP,"feature_settings.txt"));
    }


    static  void test4() throws Exception{
        ClfDataSet clfDataSet = new SparseClfDataSet(5,3,false,6);
        clfDataSet.setFeatureValue(0,0,3.5);
        clfDataSet.setFeatureValue(1,2,5.5);
        clfDataSet.setFeatureValue(4,1,2.5);
        clfDataSet.setFeatureValue(4,2,5.5);
        clfDataSet.setLabel(0, 1);
        clfDataSet.setLabel(1,2);
        clfDataSet.setLabel(2,3);
        clfDataSet.setLabel(3,5);
        clfDataSet.setLabel(4,2);
        clfDataSet.getRow(0).putSetting(new DataSetting().setExtId("zero").setExtLabel("spam"));
        clfDataSet.getRow(1).putSetting(new DataSetting().setExtId("first").setExtLabel("non-spam"));
        clfDataSet.getRow(2).putSetting(new DataSetting().setExtId("second").setExtLabel("good"));
        clfDataSet.getRow(3).putSetting(new DataSetting().setExtId("third").setExtLabel("bad"));
        clfDataSet.getRow(4).putSetting(new DataSetting().setExtId("fourth").setExtLabel("iii"));
        clfDataSet.getColumn(0).putSetting(new FeatureSetting().setFeatureName("color").setFeatureType(FeatureType.BINARY));
        clfDataSet.getColumn(1).putSetting(new FeatureSetting().setFeatureName("age").setFeatureType(FeatureType.NUMERICAL));
        clfDataSet.getColumn(2).putSetting(new FeatureSetting().setFeatureName("income").setFeatureType(FeatureType.NUMERICAL));
        DataSetUtil.dumpDataSettings(clfDataSet,new File(TMP,"datasettings.txt"));
        DataSetUtil.dumpFeatureSettings(clfDataSet,new File(TMP,"featuresettings.txt"));
    }

    private static void test5() throws Exception{
        MultiLabelClfDataSet dataSet = TRECFormat.loadMultiLabelClfDataSet(new File(DATASETS,"spam/4labels/train.trec"),
                DataSetType.ML_CLF_DENSE,true);
        System.out.println(DataSetUtil.gatherLabels(dataSet));

    }

    private static void test6() throws Exception{
        MultiLabelClfDataSet dataSet = TRECFormat.loadMultiLabelClfDataSet(new File(DATASETS,"spam/4labels/train.trec"),
                DataSetType.ML_CLF_DENSE,true);
        ClfDataSet binary = DataSetUtil.toBinary(dataSet,2);
        System.out.println(binary);

    }


}