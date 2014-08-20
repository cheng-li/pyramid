package edu.neu.ccs.pyramid.dataset;

import edu.neu.ccs.pyramid.feature.FeatureType;

import static org.junit.Assert.*;

public class TRECFormatTest {
    public static void main(String[] args) throws Exception{
        test1();
    }

    static void test1() throws Exception{
        ClfDataSet clfDataSet = new SparseClfDataSet(5,3);
        clfDataSet.setFeatureValue(0,0,3.5);
        clfDataSet.setFeatureValue(1,2,5.5);
        clfDataSet.setFeatureValue(4,1,2.5);
        clfDataSet.setFeatureValue(4,2,5.5);
        clfDataSet.setLabel(0, 1);
        clfDataSet.setLabel(1,2);
        clfDataSet.setLabel(2,3);
        clfDataSet.setLabel(3,5);
        clfDataSet.setLabel(4,2);
        clfDataSet.putDataSetting(0,new DataSetting().setExtId("zero").setExtLabel("spam"));
        clfDataSet.putDataSetting(1,new DataSetting().setExtId("first").setExtLabel("non-spam"));
        clfDataSet.putDataSetting(2,new DataSetting().setExtId("second").setExtLabel("good"));
        clfDataSet.putDataSetting(3,new DataSetting().setExtId("third").setExtLabel("bad"));
        clfDataSet.putDataSetting(4,new DataSetting().setExtId("fourth").setExtLabel("iii"));
        clfDataSet.putFeatureSetting(0,new FeatureSetting().setFeatureName("color").setFeatureType(FeatureType.BINARY));
        clfDataSet.putFeatureSetting(1,new FeatureSetting().setFeatureName("age").setFeatureType(FeatureType.NUMERICAL));
        clfDataSet.putFeatureSetting(2,new FeatureSetting().setFeatureName("income").setFeatureType(FeatureType.NUMERICAL));
        TRECFormat.save(clfDataSet,"/Users/chengli/tmp/tmp_clfdata.trec");
        ClfDataSet clfDataSet1 = TRECFormat.loadClfDataSet("/Users/chengli/tmp/tmp_clfdata.trec",DataSetType.CLF_SPARSE);
        System.out.println(clfDataSet1);


    }

}