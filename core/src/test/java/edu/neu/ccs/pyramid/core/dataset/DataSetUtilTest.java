package edu.neu.ccs.pyramid.core.dataset;

import edu.neu.ccs.pyramid.core.configuration.Config;
import edu.neu.ccs.pyramid.core.util.Pair;
import org.apache.commons.io.FileUtils;

import java.io.File;

public class DataSetUtilTest {

    private static final Config config = new Config("config/local.properties");
    private static final String DATASETS = config.getString("input.datasets");
    private static final String TMP = config.getString("output.tmp");

    public static void main(String[] args) throws Exception{

        test8();
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
//        clfDataSet.putDataPointSetting(0,new ClfDataPointSetting().setExtId("zero").setExtLabel("spam"));
//        clfDataSet.putDataPointSetting(1,new ClfDataPointSetting().setExtId("first").setExtLabel("non-spam"));
//        clfDataSet.putDataPointSetting(2,new ClfDataPointSetting().setExtId("second").setExtLabel("good"));
//        clfDataSet.putDataPointSetting(3,new ClfDataPointSetting().setExtId("third").setExtLabel("bad"));
//        clfDataSet.putDataPointSetting(4,new ClfDataPointSetting().setExtId("fourth").setExtLabel("iii"));
//        clfDataSet.putFeatureSetting(0,new FeatureSetting().setFeatureName("color").setFeatureType(FeatureType.BINARY));
//        clfDataSet.putFeatureSetting(1,new FeatureSetting().setFeatureName("age").setFeatureType(FeatureType.NUMERICAL));
//        clfDataSet.putFeatureSetting(2,new FeatureSetting().setFeatureName("income").setFeatureType(FeatureType.NUMERICAL));
//        System.out.println(clfDataSet);
        ClfDataSet trimmed = DataSetUtil.sampleFeatures(clfDataSet, 2);
        System.out.println(trimmed);
    }

    static void test2(){
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
//        clfDataSet.putDataPointSetting(0,new ClfDataPointSetting().setExtId("zero").setExtLabel("spam"));
//        clfDataSet.putDataPointSetting(1,new ClfDataPointSetting().setExtId("first").setExtLabel("non-spam"));
//        clfDataSet.putDataPointSetting(2,new ClfDataPointSetting().setExtId("second").setExtLabel("good"));
//        clfDataSet.putDataPointSetting(3,new ClfDataPointSetting().setExtId("third").setExtLabel("bad"));
//        clfDataSet.putDataPointSetting(4,new ClfDataPointSetting().setExtId("fourth").setExtLabel("iii"));
//        clfDataSet.putFeatureSetting(0,new FeatureSetting().setFeatureName("color").setFeatureType(FeatureType.BINARY));
//        clfDataSet.putFeatureSetting(1,new FeatureSetting().setFeatureName("age").setFeatureType(FeatureType.NUMERICAL));
//        clfDataSet.putFeatureSetting(2,new FeatureSetting().setFeatureName("income").setFeatureType(FeatureType.NUMERICAL));
//        System.out.println(clfDataSet);
        System.out.println("bootstrapped sample");
        System.out.println(DataSetUtil.bootstrap(clfDataSet));

    }

    static void test3() throws Exception{
        ClfDataSet dataSet = TRECFormat.loadClfDataSet(new File(DATASETS,"/spam/trec_data/train.trec"),
                DataSetType.CLF_DENSE,true);
        DataSetUtil.dumpDataPointSettings(dataSet, new File(TMP, "data_settings.txt"));
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
//        clfDataSet.putDataPointSetting(0,new ClfDataPointSetting().setExtId("zero").setExtLabel("spam"));
//        clfDataSet.putDataPointSetting(1,new ClfDataPointSetting().setExtId("first").setExtLabel("non-spam"));
//        clfDataSet.putDataPointSetting(2,new ClfDataPointSetting().setExtId("second").setExtLabel("good"));
//        clfDataSet.putDataPointSetting(3,new ClfDataPointSetting().setExtId("third").setExtLabel("bad"));
//        clfDataSet.putDataPointSetting(4,new ClfDataPointSetting().setExtId("fourth").setExtLabel("iii"));
//        clfDataSet.putFeatureSetting(0,new FeatureSetting().setFeatureName("color").setFeatureType(FeatureType.BINARY));
//        clfDataSet.putFeatureSetting(1,new FeatureSetting().setFeatureName("age").setFeatureType(FeatureType.NUMERICAL));
//        clfDataSet.putFeatureSetting(2,new FeatureSetting().setFeatureName("income").setFeatureType(FeatureType.NUMERICAL));
//        DataSetUtil.dumpDataPointSettings(clfDataSet, new File(TMP, "datasettings.txt"));
        DataSetUtil.dumpFeatureSettings(clfDataSet,new File(TMP,"featuresettings.txt"));
    }

    private static void test5() throws Exception{
        MultiLabelClfDataSet dataSet = TRECFormat.loadMultiLabelClfDataSet(new File(DATASETS,"spam/4labels/train.trec"),
                DataSetType.ML_CLF_DENSE,true);
        System.out.println(DataSetUtil.gatherMultiLabels(dataSet));

    }

    private static void test6() throws Exception{
        MultiLabelClfDataSet dataSet = TRECFormat.loadMultiLabelClfDataSet(new File(DATASETS,"spam/4labels/train.trec"),
                DataSetType.ML_CLF_DENSE,true);
        ClfDataSet binary = DataSetUtil.toBinary(dataSet,2);
        System.out.println(binary);

    }

    static void test7(){
        ClfDataSet clfDataSet = new SparseClfDataSet(5,3,false,6);
        clfDataSet.setFeatureValue(0,0,3.5);
        clfDataSet.setFeatureValue(1,2,5.5);
        clfDataSet.setFeatureValue(2,1,-9);
        clfDataSet.setFeatureValue(3,2,2);
        clfDataSet.setFeatureValue(4,1,2.5);
        clfDataSet.setFeatureValue(4,2,5.5);
        clfDataSet.setLabel(0, 1);
        clfDataSet.setLabel(1,0);
        clfDataSet.setLabel(2,1);
        clfDataSet.setLabel(3,0);
        clfDataSet.setLabel(4,1);
//        clfDataSet.putDataPointSetting(0,new ClfDataPointSetting().setExtId("zero").setExtLabel("spam"));
//        clfDataSet.putDataPointSetting(1,new ClfDataPointSetting().setExtId("first").setExtLabel("non-spam"));
//        clfDataSet.putDataPointSetting(2,new ClfDataPointSetting().setExtId("second").setExtLabel("good"));
//        clfDataSet.putDataPointSetting(3,new ClfDataPointSetting().setExtId("third").setExtLabel("bad"));
//        clfDataSet.putDataPointSetting(4,new ClfDataPointSetting().setExtId("fourth").setExtLabel("iii"));
//        clfDataSet.putFeatureSetting(0,new FeatureSetting().setFeatureName("color").setFeatureType(FeatureType.BINARY));
//        clfDataSet.putFeatureSetting(1,new FeatureSetting().setFeatureName("age").setFeatureType(FeatureType.NUMERICAL));
//        clfDataSet.putFeatureSetting(2,new FeatureSetting().setFeatureName("income").setFeatureType(FeatureType.NUMERICAL));
        System.out.println(clfDataSet);
        Pair<ClfDataSet,ClfDataSet> trainValidation = DataSetUtil.splitToTrainValidation(clfDataSet,0.6);
        System.out.println("training set");
        System.out.println(trainValidation.getFirst());
        System.out.println("validation set");
        System.out.println(trainValidation.getSecond());
    }

    private static void test8() throws Exception{
        MultiLabelClfDataSet dataSet = TRECFormat.loadMultiLabelClfDataSet(new File(DATASETS,"20newsgroup/1/train.trec"),
                DataSetType.ML_CLF_SPARSE,true);
        System.out.println(dataSet.getMetaInfo());
        System.out.println(dataSet.getLabelTranslator());
        String str = DataSetUtil.multiLabelToBinaryString(dataSet);
        FileUtils.writeStringToFile(new File(TMP,"labels.txt"),str);
    }


}