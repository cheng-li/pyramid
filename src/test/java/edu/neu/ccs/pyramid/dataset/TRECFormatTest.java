package edu.neu.ccs.pyramid.dataset;

import edu.neu.ccs.pyramid.configuration.Config;

import java.io.File;

public class TRECFormatTest {
    private static final Config config = new Config("configs/local.config");
    private static final String DATASETS = config.getString("input.datasets");
    private static final String TMP = config.getString("output.tmp");

    public static void main(String[] args) throws Exception{
//        test1();
        test2();
    }

    static void test1() throws Exception{
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
        clfDataSet.putDataPointSetting(0,new ClfDataPointSetting().setExtId("zero").setExtLabel("spam"));
        clfDataSet.putDataPointSetting(1,new ClfDataPointSetting().setExtId("first").setExtLabel("non-spam"));
        clfDataSet.putDataPointSetting(2,new ClfDataPointSetting().setExtId("second").setExtLabel("good"));
        clfDataSet.putDataPointSetting(3,new ClfDataPointSetting().setExtId("third").setExtLabel("bad"));
        clfDataSet.putDataPointSetting(4,new ClfDataPointSetting().setExtId("fourth").setExtLabel("iii"));
        clfDataSet.putFeatureSetting(0,new FeatureSetting().setFeatureName("color").setFeatureType(FeatureType.BINARY));
        clfDataSet.putFeatureSetting(1,new FeatureSetting().setFeatureName("age").setFeatureType(FeatureType.NUMERICAL));
        clfDataSet.putFeatureSetting(2,new FeatureSetting().setFeatureName("income").setFeatureType(FeatureType.NUMERICAL));
        TRECFormat.save(clfDataSet,new File(TMP,"/tmp_clfdata.trec"));
        ClfDataSet clfDataSet1 = TRECFormat.loadClfDataSet(new File(TMP,"/tmp_clfdata.trec"),DataSetType.CLF_SPARSE,true);
        System.out.println(clfDataSet1.getMetaInfo());
        System.out.println(clfDataSet1);


    }

    /**
     * add 2 fake labels in spam data set,
     * if x=spam and x_0<0.1, also label it as 2
     * if x=spam and x_1<0.1, also label it as 3
     * @throws Exception
     */
    private static void test2() throws Exception{
        test2_train();
        test2_test();
    }

    private static void test2_train() throws Exception{

        ClfDataSet singleLabeldataSet = TRECFormat.loadClfDataSet(new File(DATASETS,"/spam/trec_data/train.trec"),
                DataSetType.CLF_DENSE, true);
        int numDataPoints = singleLabeldataSet.getNumDataPoints();
        int numFeatures = singleLabeldataSet.getNumFeatures();
        MultiLabelClfDataSet dataSet = new DenseMLClfDataSet(numDataPoints,
                numFeatures,false,4);
        int[] labels = singleLabeldataSet.getLabels();
        for (int i=0;i<numDataPoints;i++){
            dataSet.addLabel(i,labels[i]);
            if (labels[i]==1 && singleLabeldataSet.getRow(i).get(0)<0.1){
                dataSet.addLabel(i,2);
            }
            if (labels[i]==1 && singleLabeldataSet.getRow(i).get(1)<0.1){
                dataSet.addLabel(i,3);
            }
            for (int j=0;j<numFeatures;j++){
                double value = singleLabeldataSet.getRow(i).get(j);
                dataSet.setFeatureValue(i,j,value);
            }
        }


        String[] extLabels = {"non-spam","spam","spam_A","spam_B"};
        LabelTranslator labelTranslator = new LabelTranslator(extLabels);
        DataSetUtil.setLabelTranslator(dataSet, labelTranslator);

        TRECFormat.save(dataSet,new File(TMP,"/4labels/train.trec"));
        MultiLabelClfDataSet loaded = TRECFormat.loadMultiLabelClfDataSet(new File(TMP,"/4labels/train.trec"),DataSetType.ML_CLF_DENSE,true);
        System.out.println(loaded.getMetaInfo());
//        System.out.println(loaded.toString());
//        System.out.println(loaded.getMultiLabels()[2070]);
    }

    private static void test2_test() throws Exception{

        ClfDataSet singleLabeldataSet = TRECFormat.loadClfDataSet(new File(DATASETS,"/spam/trec_data/test.trec"),
                DataSetType.CLF_DENSE, true);
        int numDataPoints = singleLabeldataSet.getNumDataPoints();
        int numFeatures = singleLabeldataSet.getNumFeatures();
        MultiLabelClfDataSet dataSet = new DenseMLClfDataSet(numDataPoints,
                numFeatures,false,4);
        int[] labels = singleLabeldataSet.getLabels();
        for (int i=0;i<numDataPoints;i++){
            dataSet.addLabel(i,labels[i]);
            if (labels[i]==1 && singleLabeldataSet.getRow(i).get(0)<0.1){
                dataSet.addLabel(i,2);
            }
            if (labels[i]==1 && singleLabeldataSet.getRow(i).get(1)<0.1){
                dataSet.addLabel(i,3);
            }
            for (int j=0;j<numFeatures;j++){
                double value = singleLabeldataSet.getRow(i).get(j);
                dataSet.setFeatureValue(i,j,value);
            }
        }

        String[] extLabels = {"non-spam","spam","spam_A","spam_B"};
        LabelTranslator labelTranslator = new LabelTranslator(extLabels);
        DataSetUtil.setLabelTranslator(dataSet, labelTranslator);

        TRECFormat.save(dataSet,new File(TMP,"/4labels/test.trec"));
        MultiLabelClfDataSet loaded = TRECFormat.loadMultiLabelClfDataSet(new File(TMP,"/4labels/test.trec"),DataSetType.ML_CLF_DENSE,true);
        System.out.println(loaded.getMetaInfo());
//        System.out.println(loaded.toString());
    }


}