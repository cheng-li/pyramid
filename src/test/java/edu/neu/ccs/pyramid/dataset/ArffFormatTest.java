package edu.neu.ccs.pyramid.dataset;

import edu.neu.ccs.pyramid.configuration.Config;

import java.io.File;

import static org.junit.Assert.*;

public class ArffFormatTest {
    private static final Config config = new Config("configs/local.config");
    private static final String DATASETS = config.getString("input.datasets");
    private static final String TMP = config.getString("output.tmp");

    public static void main(String[] args) throws Exception {
        test_train();
        test_test();
    }

    private static void test_train() throws Exception{

        ClfDataSet singleLabeldataSet = TRECFormat.loadClfDataSet(new File(DATASETS,"/spam/trec_data/train.trec"),
                DataSetType.CLF_DENSE, true);
        int numDataPoints = singleLabeldataSet.getNumDataPoints();
        int numFeatures = singleLabeldataSet.getNumFeatures();
        MultiLabelClfDataSet dataSet = new DenseMLClfDataSet(numDataPoints,
                numFeatures,4);
        int[] labels = singleLabeldataSet.getLabels();
        for (int i=0;i<numDataPoints;i++){
            dataSet.addLabel(i,labels[i]);
            if (labels[i]==1 && singleLabeldataSet.getFeatureRow(i).getVector().get(0)<0.1){
                dataSet.addLabel(i,2);
            }
            if (labels[i]==1 && singleLabeldataSet.getFeatureRow(i).getVector().get(1)<0.1){
                dataSet.addLabel(i,3);
            }
            for (int j=0;j<numFeatures;j++){
                double value = singleLabeldataSet.getFeatureRow(i).getVector().get(j);
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

    private static void test_test() throws Exception{

        ClfDataSet singleLabeldataSet = TRECFormat.loadClfDataSet(new File(DATASETS,"/spam/trec_data/test.trec"),
                DataSetType.CLF_DENSE, true);
        int numDataPoints = singleLabeldataSet.getNumDataPoints();
        int numFeatures = singleLabeldataSet.getNumFeatures();
        MultiLabelClfDataSet dataSet = new DenseMLClfDataSet(numDataPoints,
                numFeatures,4);
        int[] labels = singleLabeldataSet.getLabels();
        for (int i=0;i<numDataPoints;i++){
            dataSet.addLabel(i,labels[i]);
            if (labels[i]==1 && singleLabeldataSet.getFeatureRow(i).getVector().get(0)<0.1){
                dataSet.addLabel(i,2);
            }
            if (labels[i]==1 && singleLabeldataSet.getFeatureRow(i).getVector().get(1)<0.1){
                dataSet.addLabel(i,3);
            }
            for (int j=0;j<numFeatures;j++){
                double value = singleLabeldataSet.getFeatureRow(i).getVector().get(j);
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
        System.out.println(dataSet.getFeatureRow(0).getSetting());
    }
}