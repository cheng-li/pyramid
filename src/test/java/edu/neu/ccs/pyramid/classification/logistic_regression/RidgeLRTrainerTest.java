package edu.neu.ccs.pyramid.classification.logistic_regression;

import edu.neu.ccs.pyramid.classification.boosting.lktb.LKTBConfig;
import edu.neu.ccs.pyramid.classification.boosting.lktb.LKTreeBoost;
import edu.neu.ccs.pyramid.configuration.Config;
import edu.neu.ccs.pyramid.dataset.ClfDataSet;
import edu.neu.ccs.pyramid.dataset.DataSetType;
import edu.neu.ccs.pyramid.dataset.TRECFormat;
import edu.neu.ccs.pyramid.eval.Accuracy;
import org.apache.commons.lang3.time.StopWatch;

import java.io.File;
import java.util.List;

import static org.junit.Assert.*;

public class RidgeLRTrainerTest {
    private static final Config config = new Config("configs/local.config");
    private static final String DATASETS = config.getString("input.datasets");
    private static final String TMP = config.getString("output.tmp");

    public static void main(String[] args) throws Exception{

        spam_all();
    }


    static void spam_all() throws Exception{
        spam_build();
        spam_test();
    }

    static void spam_build() throws Exception{


        ClfDataSet dataSet = TRECFormat.loadClfDataSet(new File(DATASETS, "/spam/trec_data/train.trec"),
                DataSetType.CLF_DENSE, true);
        System.out.println(dataSet.getMetaInfo());

        StopWatch stopWatch = new StopWatch();
        stopWatch.start();
        LogisticRegression logisticRegression = RidgeLRTrainer.train(dataSet,10);
        System.out.println(logisticRegression.weights);
        stopWatch.stop();
        System.out.println(stopWatch);

//        for (int i=0;i<dataSet.getNumDataPoints();i++){
//            System.out.println(logisticRegression.predict(dataSet.getRow(i)));
//        }

        System.out.println("accuracy on training set ="+Accuracy.accuracy(logisticRegression,dataSet));
        logisticRegression.serialize(new File(TMP,"logistic_regression.ser"));
    }

    static void spam_test() throws Exception{


        ClfDataSet dataSet = TRECFormat.loadClfDataSet(new File(DATASETS, "/spam/trec_data/test.trec"),
                DataSetType.CLF_DENSE, true);
        LogisticRegression logisticRegression = LogisticRegression.deserialize(new File(TMP,"logistic_regression.ser"));
        System.out.println("accuracy on test set = "+Accuracy.accuracy(logisticRegression,dataSet));
    }

}