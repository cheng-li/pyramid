package edu.neu.ccs.pyramid.classification.boosting.naive_bayes;

import edu.neu.ccs.pyramid.classification.Classifier;
import edu.neu.ccs.pyramid.classification.naive_bayes.*;
import edu.neu.ccs.pyramid.configuration.Config;
import edu.neu.ccs.pyramid.dataset.ClfDataSet;
import edu.neu.ccs.pyramid.dataset.DataSetType;
import edu.neu.ccs.pyramid.dataset.TRECFormat;
import edu.neu.ccs.pyramid.eval.AUC;
import edu.neu.ccs.pyramid.eval.Accuracy;


import java.io.File;
import java.io.IOException;
import java.lang.reflect.InvocationTargetException;

/**
 * Created by Rainicy on 10/3/14.
 */
public class HistogramNBTest {
    private static final Config config = new Config("configs/local.config");
    private static final String DATASETS = config.getString("input.datasets");
    private static final String TMP = config.getString("output.tmp");

    public static void main(String[] args) throws Exception {

//        histogramNBTest(100);
//        gassianNBTest();
//        gammaNBTest();
//        bernoulliNBTest();
        multinomialNBTest();

    }

    private static void multinomialNBTest() throws IOException, ClassNotFoundException, InvocationTargetException, NoSuchMethodException, InstantiationException, IllegalAccessException {
        ClfDataSet dataSet = TRECFormat.loadClfDataSet(new File(DATASETS, "/spam/trec_data/train.trec"),
                DataSetType.CLF_DENSE, true);

        ClfDataSet testDataset = TRECFormat.loadClfDataSet(new File(DATASETS, "/spam/trec_data/test.trec"),
                DataSetType.CLF_DENSE, true);

        NaiveBayes<Multinomial> naiveBayes = new NaiveBayes<>(Multinomial.class);
        naiveBayes.build(dataSet);

        double accuracy = Accuracy.accuracy(naiveBayes,dataSet);
        System.out.println("Accuracy on training set: " + accuracy);
        System.out.println("auc on training set ="+ AUC.auc(naiveBayes,dataSet));

        accuracy = Accuracy.accuracy(naiveBayes, testDataset);
        System.out.println("Accuracy on testing set: " + accuracy);
        System.out.println("auc on test set ="+ AUC.auc(naiveBayes,testDataset));
        System.out.println();
    }

    private static void bernoulliNBTest() throws IOException, ClassNotFoundException, InvocationTargetException, NoSuchMethodException, InstantiationException, IllegalAccessException {
        ClfDataSet dataSet = TRECFormat.loadClfDataSet(new File(DATASETS, "/spam/trec_data/train.trec"),
                DataSetType.CLF_DENSE, true);

        ClfDataSet testDataset = TRECFormat.loadClfDataSet(new File(DATASETS, "/spam/trec_data/test.trec"),
                DataSetType.CLF_DENSE, true);

        NaiveBayes<Bernoulli> naiveBayes = new NaiveBayes<>(Bernoulli.class);
        naiveBayes.build(dataSet);

        double accuracy = Accuracy.accuracy(naiveBayes,dataSet);
        System.out.println("Accuracy on training set: " + accuracy);
        System.out.println("auc on training set ="+ AUC.auc(naiveBayes,dataSet));

        accuracy = Accuracy.accuracy(naiveBayes, testDataset);
        System.out.println("Accuracy on testing set: " + accuracy);
        System.out.println("auc on test set ="+ AUC.auc(naiveBayes,testDataset));
        System.out.println();

    }

    private static void gammaNBTest() throws IOException, ClassNotFoundException, InvocationTargetException, NoSuchMethodException, InstantiationException, IllegalAccessException {
        ClfDataSet dataSet = TRECFormat.loadClfDataSet(new File(DATASETS, "/spam/trec_data/train.trec"),
                DataSetType.CLF_DENSE, true);

        ClfDataSet testDataset = TRECFormat.loadClfDataSet(new File(DATASETS, "/spam/trec_data/test.trec"),
                DataSetType.CLF_DENSE, true);

        NaiveBayes<Gamma> naiveBayes = new NaiveBayes<>(Gamma.class);
        naiveBayes.build(dataSet);

        double accuracy = Accuracy.accuracy(naiveBayes,dataSet);
        System.out.println("Accuracy on training set: " + accuracy);
        System.out.println("auc on training set ="+ AUC.auc(naiveBayes,dataSet));

        accuracy = Accuracy.accuracy(naiveBayes, testDataset);
        System.out.println("Accuracy on testing set: " + accuracy);
        System.out.println("auc on test set ="+ AUC.auc(naiveBayes,testDataset));
        System.out.println();

    }


    private static void gassianNBTest() throws IOException, ClassNotFoundException, InvocationTargetException, NoSuchMethodException, InstantiationException, IllegalAccessException {
        ClfDataSet dataSet = TRECFormat.loadClfDataSet(new File(DATASETS, "/spam/trec_data/train.trec"),
                DataSetType.CLF_DENSE, true);

        ClfDataSet testDataset = TRECFormat.loadClfDataSet(new File(DATASETS, "/spam/trec_data/test.trec"),
                DataSetType.CLF_DENSE, true);

        NaiveBayes<Gaussian> naiveBayes = new NaiveBayes<>(Gaussian.class);
        naiveBayes.build(dataSet);

        double accuracy = Accuracy.accuracy(naiveBayes,dataSet);
        System.out.println("Accuracy on training set: " + accuracy);
        System.out.println("auc on training set ="+ AUC.auc(naiveBayes,dataSet));

        accuracy = Accuracy.accuracy(naiveBayes, testDataset);
        System.out.println("Accuracy on testing set: " + accuracy);
        System.out.println("auc on test set ="+ AUC.auc(naiveBayes,testDataset));
        System.out.println();
    }

    protected static void histogramNBTest(int maxBins) throws IOException, ClassNotFoundException, InvocationTargetException, NoSuchMethodException, InstantiationException, IllegalAccessException {
        ClfDataSet dataSet = TRECFormat.loadClfDataSet(new File(DATASETS, "/spam/trec_data/train.trec"),
                DataSetType.CLF_DENSE, true);

        ClfDataSet testDataset = TRECFormat.loadClfDataSet(new File(DATASETS, "/spam/trec_data/test.trec"),
                DataSetType.CLF_DENSE, true);

        for (int bins=1; bins<maxBins; bins++) {

            NaiveBayes<Histogram> naiveBayes = new NaiveBayes<>(Histogram.class);
            naiveBayes.build(dataSet, bins);

            double accuracy = Accuracy.accuracy(naiveBayes,dataSet);
            System.out.println("#Bins: " + bins + " Accuracy on training set: " + accuracy);
            System.out.println("auc on training set ="+ AUC.auc(naiveBayes,dataSet));

            accuracy = Accuracy.accuracy(naiveBayes, testDataset);
            System.out.println("#Bins: " + bins + " Accuracy on testing set: " + accuracy);
            System.out.println("auc on test set ="+ AUC.auc(naiveBayes,testDataset));
            System.out.println();
        }

    }

}
