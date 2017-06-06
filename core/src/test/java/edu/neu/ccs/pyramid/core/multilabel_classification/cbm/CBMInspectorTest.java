package edu.neu.ccs.pyramid.core.multilabel_classification.cbm;


import edu.neu.ccs.pyramid.core.configuration.Config;
import edu.neu.ccs.pyramid.core.dataset.DataSetType;
import edu.neu.ccs.pyramid.core.dataset.MultiLabel;
import edu.neu.ccs.pyramid.core.dataset.MultiLabelClfDataSet;
import edu.neu.ccs.pyramid.core.dataset.TRECFormat;
import edu.neu.ccs.pyramid.core.eval.Accuracy;
import edu.neu.ccs.pyramid.core.util.Serialization;

import java.io.File;

public class CBMInspectorTest {
    private static final Config config = new Config("config/local.properties");
    private static final String DATASETS = config.getString("input.datasets");
    private static final String TMP = config.getString("output.tmp");
    public static void main(String[] args) throws Exception{
        test1();
    }

    private static void test1() throws Exception{
        MultiLabelClfDataSet testSet = TRECFormat.loadMultiLabelClfDataSet(new File(DATASETS, "meka_imdb/1/data_sets/test"),
                DataSetType.ML_CLF_SPARSE, true);
        CBM CBM = (CBM) Serialization.deserialize(new File(TMP,"model"));
        System.out.println(Accuracy.accuracy(CBM,testSet));
        for (int i=0;i<testSet.getNumDataPoints();i++){
            MultiLabel trueLabel = testSet.getMultiLabels()[i];
            MultiLabel pred = CBM.predict(testSet.getRow(i));
            MultiLabel expectation = CBM.predictByMarginals(testSet.getRow(i));
            if (pred.equals(trueLabel)&&!pred.equals(expectation)&&expectation.getMatchedLabels().size()>0){
                System.out.println("==============================");
                System.out.println("data point "+i);
                System.out.println("prediction = "+pred);
                System.out.println("expectation = "+expectation);
                CBMInspector.covariance(CBM,testSet.getRow(i),testSet.getLabelTranslator());
            }
        }


    }

}