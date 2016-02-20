package edu.neu.ccs.pyramid.multilabel_classification.bmm_variant;


import edu.neu.ccs.pyramid.configuration.Config;
import edu.neu.ccs.pyramid.dataset.DataSetType;
import edu.neu.ccs.pyramid.dataset.MultiLabel;
import edu.neu.ccs.pyramid.dataset.MultiLabelClfDataSet;
import edu.neu.ccs.pyramid.dataset.TRECFormat;
import edu.neu.ccs.pyramid.eval.Accuracy;
import edu.neu.ccs.pyramid.util.Serialization;

import java.io.File;

public class BMMInspectorTest {
    private static final Config config = new Config("config/local.config");
    private static final String DATASETS = config.getString("input.datasets");
    private static final String TMP = config.getString("output.tmp");
    public static void main(String[] args) throws Exception{
        test1();
    }

    private static void test1() throws Exception{
        MultiLabelClfDataSet testSet = TRECFormat.loadMultiLabelClfDataSet(new File(DATASETS, "meka_imdb/1/data_sets/test"),
                DataSetType.ML_CLF_SPARSE, true);
        BMMClassifier bmmClassifier = (BMMClassifier)Serialization.deserialize(new File(TMP,"model"));
        System.out.println(Accuracy.accuracy(bmmClassifier,testSet));
        for (int i=0;i<testSet.getNumDataPoints();i++){
            MultiLabel trueLabel = testSet.getMultiLabels()[i];
            MultiLabel pred = bmmClassifier.predict(testSet.getRow(i));
            MultiLabel expectation = bmmClassifier.predictByExpectation(testSet.getRow(i));
            if (pred.equals(trueLabel)&&!pred.equals(expectation)&&expectation.getMatchedLabels().size()>0){
                System.out.println("==============================");
                System.out.println("data point "+i);
                System.out.println("prediction = "+pred);
                System.out.println("expectation = "+expectation);
                BMMInspector.covariance(bmmClassifier,testSet.getRow(i),testSet.getLabelTranslator());
            }
        }


    }

}