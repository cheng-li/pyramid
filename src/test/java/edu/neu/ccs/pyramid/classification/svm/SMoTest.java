package edu.neu.ccs.pyramid.classification.svm;

import edu.neu.ccs.pyramid.configuration.Config;
import edu.neu.ccs.pyramid.dataset.ClfDataSet;
import edu.neu.ccs.pyramid.dataset.DataSetType;
import edu.neu.ccs.pyramid.dataset.TRECFormat;
import edu.neu.ccs.pyramid.eval.AUC;
import edu.neu.ccs.pyramid.eval.Accuracy;

import java.io.File;
import java.io.IOException;
import java.util.Arrays;

/**
 * Created by Rainicy on 11/28/14.
 */
public class SMoTest {

    private static final Config config = new Config("config/local.config");
    private static final String DATASETS = config.getString("input.datasets");
    private static final String TMP = config.getString("output.tmp");

    public static void main(String[] args) throws Exception {

        smoTest();
    }

    private static void smoTest() throws IOException, ClassNotFoundException {
        ClfDataSet dataSet = TRECFormat.loadClfDataSet(new File("/home/bingyu/spame/trec_data/train.trec"),
                DataSetType.CLF_DENSE, true);

        for (int i=0; i<dataSet.getNumDataPoints(); i++) {
            if (dataSet.getLabels()[i] == 0) {
                dataSet.setLabel(i, -1);
            }
        }

        ClfDataSet testDataset = TRECFormat.loadClfDataSet(new File("/home/bingyu/spame/trec_data/test.trec"),
                DataSetType.CLF_DENSE, true);
        for (int i=0; i<testDataset.getNumDataPoints(); i++) {
            if (testDataset.getLabels()[i] == 0) {
                testDataset.setLabel(i, -1);
            }
        }
//        System.out.println(Arrays.toString(testDataset.getLabels()));

        System.out.println(dataSet.getMetaInfo());
        SMO smo = new SMO(0.05, 0.0001, 0.0001, 50, "linear");
        smo.train(dataSet);
        int[] resulst = smo.predict(dataSet);
        System.out.println(Arrays.toString(resulst));
        double accuracy = Accuracy.accuracy(smo, dataSet);
        System.out.println("Accuracy on training set: " + accuracy);

        accuracy = Accuracy.accuracy(smo, testDataset);
        System.out.println("Accuracy on testing set: " + accuracy);
        System.out.println();
    }
}
