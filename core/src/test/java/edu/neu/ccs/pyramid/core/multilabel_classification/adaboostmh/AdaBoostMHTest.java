package edu.neu.ccs.pyramid.core.multilabel_classification.adaboostmh;

import edu.neu.ccs.pyramid.core.configuration.Config;
import edu.neu.ccs.pyramid.core.dataset.DataSetType;
import edu.neu.ccs.pyramid.core.dataset.MultiLabelClfDataSet;
import edu.neu.ccs.pyramid.core.dataset.TRECFormat;
import edu.neu.ccs.pyramid.core.eval.Accuracy;
import edu.neu.ccs.pyramid.core.eval.Overlap;
import org.apache.commons.lang3.time.StopWatch;

import java.io.File;

public class AdaBoostMHTest {
    private static final Config config = new Config("config/local.properties");
    private static final String DATASETS = config.getString("input.datasets");
    private static final String TMP = config.getString("output.tmp");

    public static void main(String[] args) throws Exception{
        test1();

    }

    static void test1() throws Exception{
        MultiLabelClfDataSet dataSet = TRECFormat.loadMultiLabelClfDataSet(new File(DATASETS, "ohsumed/3/train.trec"), DataSetType.ML_CLF_SPARSE, true);
        MultiLabelClfDataSet testSet = TRECFormat.loadMultiLabelClfDataSet(new File(DATASETS, "ohsumed/3/test.trec"), DataSetType.ML_CLF_SPARSE, true);
        AdaBoostMH boosting = new AdaBoostMH(dataSet.getNumClasses());

        AdaBoostMHTrainer trainer = new AdaBoostMHTrainer(dataSet,boosting);

        StopWatch stopWatch = new StopWatch();
        stopWatch.start();
        for (int round =0;round<500;round++){
            System.out.println("round="+round);
            trainer.iterate();
            System.out.println(stopWatch);
        }

        System.out.println("training accuracy="+ Accuracy.accuracy(boosting, dataSet));
        System.out.println("training overlap = "+ Overlap.overlap(boosting,dataSet));
        System.out.println("test accuracy="+ Accuracy.accuracy(boosting, testSet));
        System.out.println("test overlap = "+ Overlap.overlap(boosting,testSet));
    }

}