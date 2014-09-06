package edu.neu.ccs.pyramid.dataset;

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
        clfDataSet.getFeatureRow(0).putSetting(new DataSetting().setExtId("zero").setExtLabel("spam"));
        clfDataSet.getFeatureRow(1).putSetting(new DataSetting().setExtId("first").setExtLabel("non-spam"));
        clfDataSet.getFeatureRow(2).putSetting(new DataSetting().setExtId("second").setExtLabel("good"));
        clfDataSet.getFeatureRow(3).putSetting(new DataSetting().setExtId("third").setExtLabel("bad"));
        clfDataSet.getFeatureRow(4).putSetting(new DataSetting().setExtId("fourth").setExtLabel("iii"));
        clfDataSet.getFeatureColumn(0).putSetting(new FeatureSetting().setFeatureName("color").setFeatureType(FeatureType.BINARY));
        clfDataSet.getFeatureColumn(1).putSetting(new FeatureSetting().setFeatureName("age").setFeatureType(FeatureType.NUMERICAL));
        clfDataSet.getFeatureColumn(2).putSetting(new FeatureSetting().setFeatureName("income").setFeatureType(FeatureType.NUMERICAL));
        TRECFormat.save(clfDataSet,"/Users/chengli/tmp/tmp_clfdata.trec");
        ClfDataSet clfDataSet1 = TRECFormat.loadClfDataSet("/Users/chengli/tmp/tmp_clfdata.trec",DataSetType.CLF_SPARSE,true);
        System.out.println(clfDataSet1);


    }

}