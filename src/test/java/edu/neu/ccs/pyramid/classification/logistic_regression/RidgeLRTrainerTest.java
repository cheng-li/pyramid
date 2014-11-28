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
        spam_build();
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

        System.out.println("accuracy="+Accuracy.accuracy(logisticRegression,dataSet));


    }

}