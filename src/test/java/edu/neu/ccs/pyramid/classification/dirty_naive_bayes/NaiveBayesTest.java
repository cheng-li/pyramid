package edu.neu.ccs.pyramid.classification.dirty_naive_bayes;

import edu.neu.ccs.pyramid.configuration.Config;
import edu.neu.ccs.pyramid.dataset.ClfDataSet;
import edu.neu.ccs.pyramid.dataset.DataSetType;
import edu.neu.ccs.pyramid.dataset.TRECFormat;
import edu.neu.ccs.pyramid.eval.Accuracy;

import java.io.File;

public class NaiveBayesTest {
    private static final Config config = new Config("configs/local.config");
    private static final String DATASETS = config.getString("input.datasets");
    private static final String TMP = config.getString("output.tmp");
    public static void main(String[] args) throws Exception{
        test1();

    }

    private static void test1() throws Exception{
        ClfDataSet dataSet = TRECFormat.loadClfDataSet(new File(DATASETS, "20newsgroup/1/train.trec"),
                DataSetType.CLF_SPARSE, true);

        ClfDataSet testDataset = TRECFormat.loadClfDataSet(new File(DATASETS, "20newsgroup/1/test.trec"),
                DataSetType.CLF_SPARSE, true);
        System.out.println(dataSet.getMetaInfo());
        System.out.println("test");
        System.out.println(testDataset.getMetaInfo());
        System.out.println("start training");
        NaiveBayes naiveBayes = NBTrainer.train(dataSet);
        System.out.println("training done");
        System.out.println(Accuracy.accuracy(naiveBayes,dataSet));
        System.out.println(Accuracy.accuracy(naiveBayes,testDataset));
    }

}