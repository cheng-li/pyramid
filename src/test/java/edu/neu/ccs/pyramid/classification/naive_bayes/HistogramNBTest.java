package edu.neu.ccs.pyramid.classification.naive_bayes;

import com.google.common.base.Stopwatch;
import edu.neu.ccs.pyramid.configuration.Config;
import edu.neu.ccs.pyramid.dataset.ClfDataSet;
import edu.neu.ccs.pyramid.dataset.DataSetType;
import edu.neu.ccs.pyramid.dataset.TRECFormat;
import edu.neu.ccs.pyramid.eval.AUC;
import edu.neu.ccs.pyramid.eval.Accuracy;

import java.io.*;
import java.lang.reflect.InvocationTargetException;

/**
 * Created by Rainicy on 10/3/14...
 */
public class HistogramNBTest {
    private static final Config config = new Config("configs/local.config");
    private static final String DATASETS = config.getString("input.datasets");
    private static final String TMP = config.getString("output.tmp");

    public static void main(String[] args) throws Exception {
        System.out.println(config.toString());

//        histogramNBTest(10);
//        gassianNBTest();
//        gammaNBTest();
//        bernoulliNBTest();
//        multinomialNBTest();
//        cnnTest();

//        localBernoulliTest();
        // tests for Spam
//        localBernoulliSpamTest();
        localGaussianSpamTest();
//        localHistSpamTest(100);
//        localMulSpamTest();

        // tests for 20newsgroup
//        localBernoulli20Test();
//        localHist20Test(5);
//        localMul20Test();
//        localGaussian20Test();

//        localBernoulliIMDBTest();
//        localHistIMDBTest(5);
//        localMulIMDBTest();
//        localGaussianIMDBTest();
//
//        localBernoulliBabyTest();
//        localHistBabyTest(5);
//        localMulBabyTest();
//        localGaussianBabyTest();
    }

    private static void localBernoulli20Test() throws IOException, ClassNotFoundException, InvocationTargetException, NoSuchMethodException, InstantiationException, IllegalAccessException {
        System.out.println("------------Bernoulli 20Groups-------------");
        ClfDataSet dataSet = TRECFormat.loadClfDataSet(new File(DATASETS, "20newsgroup/1/train.trec"),
                DataSetType.CLF_SPARSE, true);
        ClfDataSet testDataset = TRECFormat.loadClfDataSet(new File(DATASETS, "20newsgroup/1/test.trec"),
                DataSetType.CLF_SPARSE, true);

        System.out.println(dataSet.getMetaInfo());
        System.out.println("Starting training ...");

        Stopwatch stopwatch = Stopwatch.createStarted();
        NaiveBayes<Bernoulli> naiveBayes = new NaiveBayes<>(Bernoulli.class);
        naiveBayes.build(dataSet);
        System.out.println("Training time: " + stopwatch.stop());
        System.out.println("training done...");

        System.out.println();

        System.out.println("starting predicting ...");
        stopwatch.reset();
        stopwatch.start();
        double accuracy = Accuracy.accuracy(naiveBayes,dataSet);
        System.out.println("Accuracy on training set: " + accuracy);
        System.out.println("Prediction time: " + stopwatch.stop());

        System.out.println();

        System.out.println("starting predicting ...");
        stopwatch.reset();
        stopwatch.start();
        accuracy = Accuracy.accuracy(naiveBayes, testDataset);
        System.out.println("Accuracy on testing set: " + accuracy);
        System.out.println("Prediction time: " + stopwatch.stop());
        System.out.println();
    }

    private static void localHist20Test(int maxBins) throws IOException, ClassNotFoundException, InvocationTargetException, NoSuchMethodException, InstantiationException, IllegalAccessException {
        System.out.println("------------Hist 20Groups-------------");
        ClfDataSet dataSet = TRECFormat.loadClfDataSet(new File(DATASETS, "20newsgroup/1/train.trec"),
                DataSetType.CLF_SPARSE, true);
        ClfDataSet testDataset = TRECFormat.loadClfDataSet(new File(DATASETS, "20newsgroup/1/test.trec"),
                DataSetType.CLF_SPARSE, true);

        System.out.println(dataSet.getMetaInfo());
        System.out.println("Starting training ...");

        Stopwatch stopwatch = Stopwatch.createStarted();
        NaiveBayes<Histogram> naiveBayes = new NaiveBayes<>(Histogram.class);
        naiveBayes.build(dataSet, maxBins);
        System.out.println("Training time: " + stopwatch.stop());
        System.out.println("training done...");

        System.out.println();

        System.out.println("starting predicting ...");
        stopwatch.reset();
        stopwatch.start();
        double accuracy = Accuracy.accuracy(naiveBayes,dataSet);
        System.out.println("Accuracy on training set: " + accuracy);
        System.out.println("Prediction time: " + stopwatch.stop());

        System.out.println();

        System.out.println("starting predicting ...");
        stopwatch.reset();
        stopwatch.start();
        accuracy = Accuracy.accuracy(naiveBayes, testDataset);
        System.out.println("Accuracy on testing set: " + accuracy);
        System.out.println("Prediction time: " + stopwatch.stop());
        System.out.println();
    }

    private static void localMul20Test() throws IOException, ClassNotFoundException, InvocationTargetException, NoSuchMethodException, InstantiationException, IllegalAccessException {
        System.out.println("------------Mulnomial 20Groups-------------");
        ClfDataSet dataSet = TRECFormat.loadClfDataSet(new File(DATASETS, "20newsgroup/1/train.trec"),
                DataSetType.CLF_SPARSE, true);
        ClfDataSet testDataset = TRECFormat.loadClfDataSet(new File(DATASETS, "20newsgroup/1/test.trec"),
                DataSetType.CLF_SPARSE, true);

        System.out.println(dataSet.getMetaInfo());
        System.out.println("Starting training ...");

        Stopwatch stopwatch = Stopwatch.createStarted();
        NaiveBayes<Multinomial> naiveBayes = new NaiveBayes<>(Multinomial.class);
        naiveBayes.build(dataSet);
        System.out.println("Training time: " + stopwatch.stop());
        System.out.println("training done...");

        System.out.println();

        System.out.println("starting predicting ...");
        stopwatch.reset();
        stopwatch.start();
        double accuracy = Accuracy.accuracy(naiveBayes,dataSet);
        System.out.println("Accuracy on training set: " + accuracy);
        System.out.println("Prediction time: " + stopwatch.stop());

        System.out.println();

        System.out.println("starting predicting ...");
        stopwatch.reset();
        stopwatch.start();
        accuracy = Accuracy.accuracy(naiveBayes, testDataset);
        System.out.println("Accuracy on testing set: " + accuracy);
        System.out.println("Prediction time: " + stopwatch.stop());
        System.out.println();
    }

    private static void localGaussian20Test() throws IOException, ClassNotFoundException, InvocationTargetException, NoSuchMethodException, InstantiationException, IllegalAccessException {
        System.out.println("------------Gaussian 20Groups-------------");
        ClfDataSet dataSet = TRECFormat.loadClfDataSet(new File(DATASETS, "20newsgroup/1/train.trec"),
                DataSetType.CLF_SPARSE, true);
        ClfDataSet testDataset = TRECFormat.loadClfDataSet(new File(DATASETS, "20newsgroup/1/test.trec"),
                DataSetType.CLF_SPARSE, true);

        System.out.println(dataSet.getMetaInfo());
        System.out.println("Starting training ...");

        Stopwatch stopwatch = Stopwatch.createStarted();
        NaiveBayes<Gaussian> naiveBayes = new NaiveBayes<>(Gaussian.class);
        naiveBayes.build(dataSet);
        System.out.println("Training time: " + stopwatch.stop());
        System.out.println("training done...");

        System.out.println();

        System.out.println("starting predicting ...");
        stopwatch.reset();
        stopwatch.start();
        double accuracy = Accuracy.accuracy(naiveBayes,dataSet);
        System.out.println("Accuracy on training set: " + accuracy);
        System.out.println("Prediction time: " + stopwatch.stop());

        System.out.println();

        System.out.println("starting predicting ...");
        stopwatch.reset();
        stopwatch.start();
        accuracy = Accuracy.accuracy(naiveBayes, testDataset);
        System.out.println("Accuracy on testing set: " + accuracy);
        System.out.println("Prediction time: " + stopwatch.stop());
        System.out.println();
    }

    private static void localBernoulliSpamTest() throws IOException, ClassNotFoundException, InvocationTargetException, NoSuchMethodException, InstantiationException, IllegalAccessException {
        System.out.println("------------Bernoulli Spam-------------");

        ClfDataSet dataSet = TRECFormat.loadClfDataSet(new File(DATASETS, "spam/trec_data/train.trec"),
                DataSetType.CLF_SPARSE, true);
        ClfDataSet testDataset = TRECFormat.loadClfDataSet(new File(DATASETS, "spam/trec_data/test.trec"),
                DataSetType.CLF_SPARSE, true);

        System.out.println(dataSet.getMetaInfo());
        System.out.println("Starting training ...");

        Stopwatch stopwatch = Stopwatch.createStarted();
        NaiveBayes<Bernoulli> naiveBayes = new NaiveBayes<>(Bernoulli.class);
        naiveBayes.build(dataSet);
        System.out.println("Training time: " + stopwatch.stop());
        System.out.println("training done...");

        System.out.println();

        System.out.println("starting predicting ...");
        stopwatch.reset();
        stopwatch.start();
        double accuracy = Accuracy.accuracy(naiveBayes,dataSet);
        System.out.println("Accuracy on training set: " + accuracy);
        System.out.println("Prediction time: " + stopwatch.stop());

        System.out.println();

        System.out.println("starting predicting ...");
        stopwatch.reset();
        stopwatch.start();
        accuracy = Accuracy.accuracy(naiveBayes, testDataset);
        System.out.println("Accuracy on testing set: " + accuracy);
        System.out.println("Prediction time: " + stopwatch.stop());
        System.out.println();
    }

    private static void localGaussianSpamTest() throws IOException, ClassNotFoundException, InvocationTargetException, NoSuchMethodException, InstantiationException, IllegalAccessException {
        System.out.println("------------Gaussian Spam-------------");

        ClfDataSet dataSet = TRECFormat.loadClfDataSet(new File(DATASETS, "spam/trec_data/train.trec"),
                DataSetType.CLF_SPARSE, true);
        ClfDataSet testDataset = TRECFormat.loadClfDataSet(new File(DATASETS, "spam/trec_data/test.trec"),
                DataSetType.CLF_SPARSE, true);

        System.out.println(dataSet.getMetaInfo());
        System.out.println("Starting training ...");

        Stopwatch stopwatch = Stopwatch.createStarted();
        NaiveBayes<Gaussian> naiveBayes = new NaiveBayes<>(Gaussian.class);
        naiveBayes.build(dataSet);
        System.out.println("Training time: " + stopwatch.stop());
        System.out.println("training done...");

        System.out.println();

        System.out.println("starting predicting ...");
        stopwatch.reset();
        stopwatch.start();
        double accuracy = Accuracy.accuracy(naiveBayes,dataSet);
        System.out.println("Accuracy on training set: " + accuracy);
        System.out.println("Prediction time: " + stopwatch.stop());

        System.out.println();

        System.out.println("starting predicting ...");
        stopwatch.reset();
        stopwatch.start();
        accuracy = Accuracy.accuracy(naiveBayes, testDataset);
        System.out.println("Accuracy on testing set: " + accuracy);
        System.out.println("Prediction time: " + stopwatch.stop());
        System.out.println();
    }

    private static void localHistSpamTest(int maxBins) throws IOException, ClassNotFoundException, InvocationTargetException, NoSuchMethodException, InstantiationException, IllegalAccessException {
        System.out.println("------------Hist Spam-------------");
        ClfDataSet dataSet = TRECFormat.loadClfDataSet(new File(DATASETS, "spam/trec_data/train.trec"),
                DataSetType.CLF_SPARSE, true);
        ClfDataSet testDataset = TRECFormat.loadClfDataSet(new File(DATASETS, "spam/trec_data/test.trec"),
                DataSetType.CLF_SPARSE, true);

        System.out.println(dataSet.getMetaInfo());
        System.out.println("Starting training ...");

        Stopwatch stopwatch = Stopwatch.createStarted();
        NaiveBayes<Histogram> naiveBayes = new NaiveBayes<>(Histogram.class);
        naiveBayes.build(dataSet, maxBins);
        System.out.println("Training time: " + stopwatch.stop());
        System.out.println("training done...");

        System.out.println();

        System.out.println("starting predicting ...");
        stopwatch.reset();
        stopwatch.start();
        double accuracy = Accuracy.accuracy(naiveBayes,dataSet);
        System.out.println("Accuracy on training set: " + accuracy);
        System.out.println("Prediction time: " + stopwatch.stop());

        System.out.println();

        System.out.println("starting predicting ...");
        stopwatch.reset();
        stopwatch.start();
        accuracy = Accuracy.accuracy(naiveBayes, testDataset);
        System.out.println("Accuracy on testing set: " + accuracy);
        System.out.println("Prediction time: " + stopwatch.stop());
        System.out.println();
    }

    private static void localMulSpamTest() throws IOException, ClassNotFoundException, InvocationTargetException, NoSuchMethodException, InstantiationException, IllegalAccessException {
        System.out.println("------------Multinomial Spam-------------");
        ClfDataSet dataSet = TRECFormat.loadClfDataSet(new File(DATASETS, "spam/trec_data/train.trec"),
                DataSetType.CLF_SPARSE, true);
        ClfDataSet testDataset = TRECFormat.loadClfDataSet(new File(DATASETS, "spam/trec_data/test.trec"),
                DataSetType.CLF_SPARSE, true);

        System.out.println(dataSet.getMetaInfo());
        System.out.println("Starting training ...");

        Stopwatch stopwatch = Stopwatch.createStarted();
        NaiveBayes<Multinomial> naiveBayes = new NaiveBayes<>(Multinomial.class);
        naiveBayes.build(dataSet);
        System.out.println("Training time: " + stopwatch.stop());
        System.out.println("training done...");

        System.out.println();

        System.out.println("starting predicting ...");
        stopwatch.reset();
        stopwatch.start();
        double accuracy = Accuracy.accuracy(naiveBayes,dataSet);
        System.out.println("Accuracy on training set: " + accuracy);
        System.out.println("Prediction time: " + stopwatch.stop());

        System.out.println();

        System.out.println("starting predicting ...");

        stopwatch.reset();
        stopwatch.start();
        accuracy = Accuracy.accuracy(naiveBayes, testDataset);
        System.out.println("Accuracy on testing set: " + accuracy);
        System.out.println("Prediction time: " + stopwatch.stop());
        System.out.println();
    }




//
//    private static void cnnTest() throws IOException, ClassNotFoundException, InvocationTargetException, NoSuchMethodException, InstantiationException, IllegalAccessException {
//        ClfDataSet dataSet = TRECFormat.loadClfDataSet(new File("/Users/Rainicy/Datasets/pyramid/cnn/1/train.trec"),
//                DataSetType.CLF_DENSE, true);
//
//        ClfDataSet testDataset = TRECFormat.loadClfDataSet(new File("/Users/Rainicy/Datasets/pyramid/cnn/1/test.trec"),
//                DataSetType.CLF_DENSE, true);
//
//        NaiveBayes<Multinomial> naiveBayes = new NaiveBayes<>(Multinomial.class);
//        naiveBayes.build(dataSet);
//
//        double accuracy = Accuracy.accuracy(naiveBayes,dataSet);
//        System.out.println("Accuracy on training set: " + accuracy);
//        System.out.println("auc on training set ="+ AUC.auc(naiveBayes,dataSet));
//
//        accuracy = Accuracy.accuracy(naiveBayes, testDataset);
//        System.out.println("Accuracy on testing set: " + accuracy);
//        System.out.println("auc on test set ="+ AUC.auc(naiveBayes,testDataset));
//        System.out.println();
//    }
//
//    private static void multinomialNBTest() throws IOException, ClassNotFoundException, InvocationTargetException, NoSuchMethodException, InstantiationException, IllegalAccessException {
//        ClfDataSet dataSet = TRECFormat.loadClfDataSet(new File(DATASETS, "20newsgroup/1/train.trec"),
//                DataSetType.CLF_SPARSE, true);
//
//        ClfDataSet testDataset = TRECFormat.loadClfDataSet(new File(DATASETS, "20newsgroup/1/test.trec"),
//                DataSetType.CLF_SPARSE, true);
//
//        NaiveBayes<Multinomial> naiveBayes = new NaiveBayes<>(Multinomial.class);
//        naiveBayes.build(dataSet);
//
//        double accuracy = Accuracy.accuracy(naiveBayes,dataSet);
//        System.out.println("Accuracy on training set: " + accuracy);
////        System.out.println("auc on training set ="+ AUC.auc(naiveBayes,dataSet));
//
//        accuracy = Accuracy.accuracy(naiveBayes, testDataset);
//        System.out.println("Accuracy on testing set: " + accuracy);
//        System.out.println("auc on test set ="+ AUC.auc(naiveBayes,testDataset));
////        System.out.println();
//    }
//
//    private static void bernoulliNBTest() throws IOException, ClassNotFoundException, InvocationTargetException, NoSuchMethodException, InstantiationException, IllegalAccessException {
//        ClfDataSet dataSet = TRECFormat.loadClfDataSet(new File(DATASETS, "20newsgroup/1/train.trec"),
//                DataSetType.CLF_SPARSE, true);
//
//        ClfDataSet testDataset = TRECFormat.loadClfDataSet(new File(DATASETS, "20newsgroup/1/test.trec"),
//                DataSetType.CLF_SPARSE, true);
//
//        NaiveBayes<Bernoulli> naiveBayes = new NaiveBayes<>(Bernoulli.class);
//        naiveBayes.build(dataSet);
//
//        double accuracy = Accuracy.accuracy(naiveBayes,dataSet);
//        System.out.println("Accuracy on training set: " + accuracy);
////        System.out.println("auc on training set ="+ AUC.auc(naiveBayes,dataSet));
//
//        accuracy = Accuracy.accuracy(naiveBayes, testDataset);
//        System.out.println("Accuracy on testing set: " + accuracy);
////        System.out.println("auc on test set ="+ AUC.auc(naiveBayes,testDataset));
//        System.out.println();
//
//    }
//
////    private static void gammaNBTest() throws IOException, ClassNotFoundException, InvocationTargetException, NoSuchMethodException, InstantiationException, IllegalAccessException {
////        ClfDataSet dataSet = TRECFormat.loadClfDataSet(new File(DATASETS, "/spam/trec_data/train.trec"),
////                DataSetType.CLF_DENSE, true);
////
////        ClfDataSet testDataset = TRECFormat.loadClfDataSet(new File(DATASETS, "/spam/trec_data/test.trec"),
////                DataSetType.CLF_DENSE, true);
////
////        NaiveBayes<Gamma> naiveBayes = new NaiveBayes<>(Gamma.class);
////        naiveBayes.build(dataSet);
////
////        double accuracy = Accuracy.accuracy(naiveBayes,dataSet);
////        System.out.println("Accuracy on training set: " + accuracy);
////        System.out.println("auc on training set ="+ AUC.auc(naiveBayes,dataSet));
////
////        accuracy = Accuracy.accuracy(naiveBayes, testDataset);
////        System.out.println("Accuracy on testing set: " + accuracy);
////        System.out.println("auc on test set ="+ AUC.auc(naiveBayes,testDataset));
////        System.out.println();
////
////    }
//
//
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
//
    protected static void histogramNBTest(int maxBins) throws IOException, ClassNotFoundException, InvocationTargetException, NoSuchMethodException, InstantiationException, IllegalAccessException {
        ClfDataSet dataSet = TRECFormat.loadClfDataSet(new File(DATASETS, "20newsgroup/1/train.trec"),
                DataSetType.CLF_SPARSE, true);

        ClfDataSet testDataset = TRECFormat.loadClfDataSet(new File(DATASETS, "20newsgroup/1/test.trec"),
                DataSetType.CLF_SPARSE, true);

        for (int bins=1; bins<maxBins; bins++) {

            NaiveBayes<Histogram> naiveBayes = new NaiveBayes<>(Histogram.class);
            naiveBayes.build(dataSet, bins);

            double accuracy = Accuracy.accuracy(naiveBayes,dataSet);
            System.out.println("#Bins: " + bins + " Accuracy on training set: " + accuracy);
//            System.out.println("auc on training set ="+ AUC.auc(naiveBayes,dataSet));

            accuracy = Accuracy.accuracy(naiveBayes, testDataset);
            System.out.println("#Bins: " + bins + " Accuracy on testing set: " + accuracy);
//            System.out.println("auc on test set ="+ AUC.auc(naiveBayes,testDataset));
            System.out.println();
        }
    }

    private static void localBernoulliIMDBTest() throws IOException, ClassNotFoundException, InvocationTargetException, NoSuchMethodException, InstantiationException, IllegalAccessException {
        BufferedWriter bw = new BufferedWriter(new FileWriter("~/imdb_bernoulli.log"));
        System.out.println("------------Bernoulli IMDB-------------");
        bw.write("------------Bernoulli IMDB-------------\n");
        ClfDataSet dataSet = TRECFormat.loadClfDataSet(new File(DATASETS, "imdb/11/train.trec"),
                DataSetType.CLF_SPARSE, true);
        ClfDataSet testDataset = TRECFormat.loadClfDataSet(new File(DATASETS, "imdb/11/test.trec"),
                DataSetType.CLF_SPARSE, true);

        System.out.println(dataSet.getMetaInfo());
        System.out.println("Starting training ...");
        bw.write(dataSet.getMetaInfo()+"\n");
        bw.write("Starting training ...\n");

        Stopwatch stopwatch = Stopwatch.createStarted();
        NaiveBayes<Bernoulli> naiveBayes = new NaiveBayes<>(Bernoulli.class);
        naiveBayes.build(dataSet);
        System.out.println("Training time: " + stopwatch.stop());
        System.out.println("training done...");
        bw.write("Training time: " + stopwatch.stop() + "\n");
        bw.write("training done...\n");
        bw.write("\n");

        System.out.println();

        System.out.println("starting predicting ...");
        bw.write("starting predicting ...\n");
        stopwatch.reset();
        stopwatch.start();
        double accuracy = Accuracy.accuracy(naiveBayes,dataSet);
        bw.write("Accuracy on training set: " + accuracy + "\n");
        bw.write("Prediction time: " + stopwatch.stop() + "\n");
        System.out.println("Accuracy on training set: " + accuracy);
        System.out.println("Prediction time: " + stopwatch.stop());

        System.out.println();
        bw.write("\nstarting predicting ...\n");
        System.out.println("starting predicting ...");
        stopwatch.reset();
        stopwatch.start();
        accuracy = Accuracy.accuracy(naiveBayes, testDataset);
        bw.write("Accuracy on testing set: " + accuracy + "\n");
        bw.write("Prediction time: " + stopwatch.stop() + "\n\n");
        System.out.println("Accuracy on testing set: " + accuracy);
        System.out.println("Prediction time: " + stopwatch.stop());
        System.out.println();
        bw.close();
    }

    private static void localHistIMDBTest(int maxBins) throws IOException, ClassNotFoundException, InvocationTargetException, NoSuchMethodException, InstantiationException, IllegalAccessException {
        BufferedWriter bw = new BufferedWriter(new FileWriter("~/imdb_hist.log"));
        System.out.println("------------Hist IMDB-------------");
        bw.write("------------Hist IMDB-------------\n");
        ClfDataSet dataSet = TRECFormat.loadClfDataSet(new File(DATASETS, "imdb/11/train.trec"),
                DataSetType.CLF_SPARSE, true);
        ClfDataSet testDataset = TRECFormat.loadClfDataSet(new File(DATASETS, "imdb/11/test.trec"),
                DataSetType.CLF_SPARSE, true);

        System.out.println(dataSet.getMetaInfo());
        System.out.println("Starting training ...");
        bw.write(dataSet.getMetaInfo()+"\n");
        bw.write("Starting training ...\n");

        Stopwatch stopwatch = Stopwatch.createStarted();
        NaiveBayes<Histogram> naiveBayes = new NaiveBayes<>(Histogram.class);
        naiveBayes.build(dataSet, maxBins);
        System.out.println("Training time: " + stopwatch.stop());
        System.out.println("training done...");
        bw.write("Training time: " + stopwatch.stop() + "\n");
        bw.write("training done...\n");
        bw.write("\n");

        System.out.println();

        System.out.println("starting predicting ...");
        bw.write("starting predicting ...\n");
        stopwatch.reset();
        stopwatch.start();
        double accuracy = Accuracy.accuracy(naiveBayes,dataSet);
        bw.write("Accuracy on training set: " + accuracy + "\n");
        bw.write("Prediction time: " + stopwatch.stop() + "\n");
        System.out.println("Accuracy on training set: " + accuracy);
        System.out.println("Prediction time: " + stopwatch.stop());

        System.out.println();
        bw.write("\nstarting predicting ...\n");
        System.out.println("starting predicting ...");
        stopwatch.reset();
        stopwatch.start();
        accuracy = Accuracy.accuracy(naiveBayes, testDataset);
        bw.write("Accuracy on testing set: " + accuracy + "\n");
        bw.write("Prediction time: " + stopwatch.stop() + "\n\n");
        System.out.println("Accuracy on testing set: " + accuracy);
        System.out.println("Prediction time: " + stopwatch.stop());
        System.out.println();
        bw.close();
    }

    private static void localMulIMDBTest() throws IOException, ClassNotFoundException, InvocationTargetException, NoSuchMethodException, InstantiationException, IllegalAccessException {
        BufferedWriter bw = new BufferedWriter(new FileWriter("~/imdb_mul.log"));
        System.out.println("------------Multi IMDB-------------");
        bw.write("------------Multi IMDB-------------\n");
        ClfDataSet dataSet = TRECFormat.loadClfDataSet(new File(DATASETS, "imdb/11/train.trec"),
                DataSetType.CLF_SPARSE, true);
        ClfDataSet testDataset = TRECFormat.loadClfDataSet(new File(DATASETS, "imdb/11/test.trec"),
                DataSetType.CLF_SPARSE, true);

        System.out.println(dataSet.getMetaInfo());
        System.out.println("Starting training ...");
        bw.write(dataSet.getMetaInfo()+"\n");
        bw.write("Starting training ...\n");

        Stopwatch stopwatch = Stopwatch.createStarted();
        NaiveBayes<Multinomial> naiveBayes = new NaiveBayes<>(Multinomial.class);
        naiveBayes.build(dataSet);
        System.out.println("Training time: " + stopwatch.stop());
        System.out.println("training done...");
        bw.write("Training time: " + stopwatch.stop() + "\n");
        bw.write("training done...\n");
        bw.write("\n");

        System.out.println();

        System.out.println("starting predicting ...");
        bw.write("starting predicting ...\n");
        stopwatch.reset();
        stopwatch.start();
        double accuracy = Accuracy.accuracy(naiveBayes,dataSet);
        bw.write("Accuracy on training set: " + accuracy + "\n");
        bw.write("Prediction time: " + stopwatch.stop() + "\n");
        System.out.println("Accuracy on training set: " + accuracy);
        System.out.println("Prediction time: " + stopwatch.stop());

        System.out.println();
        bw.write("\nstarting predicting ...\n");
        System.out.println("starting predicting ...");
        stopwatch.reset();
        stopwatch.start();
        accuracy = Accuracy.accuracy(naiveBayes, testDataset);
        bw.write("Accuracy on testing set: " + accuracy + "\n");
        bw.write("Prediction time: " + stopwatch.stop() + "\n\n");
        System.out.println("Accuracy on testing set: " + accuracy);
        System.out.println("Prediction time: " + stopwatch.stop());
        System.out.println();
        bw.close();
    }

    private static void localGaussianIMDBTest() throws IOException, ClassNotFoundException, InvocationTargetException, NoSuchMethodException, InstantiationException, IllegalAccessException {
        BufferedWriter bw = new BufferedWriter(new FileWriter("~/imdb_gaussian.log"));
        System.out.println("------------Gaussian IMDB-------------");
        bw.write("------------Gaussian IMDB-------------\n");
        ClfDataSet dataSet = TRECFormat.loadClfDataSet(new File(DATASETS, "imdb/11/train.trec"),
                DataSetType.CLF_SPARSE, true);
        ClfDataSet testDataset = TRECFormat.loadClfDataSet(new File(DATASETS, "imdb/11/test.trec"),
                DataSetType.CLF_SPARSE, true);

        System.out.println(dataSet.getMetaInfo());
        System.out.println("Starting training ...");
        bw.write(dataSet.getMetaInfo()+"\n");
        bw.write("Starting training ...\n");

        Stopwatch stopwatch = Stopwatch.createStarted();
        NaiveBayes<Gaussian> naiveBayes = new NaiveBayes<>(Gaussian.class);
        naiveBayes.build(dataSet);
        System.out.println("Training time: " + stopwatch.stop());
        System.out.println("training done...");
        bw.write("Training time: " + stopwatch.stop() + "\n");
        bw.write("training done...\n");
        bw.write("\n");

        System.out.println();

        System.out.println("starting predicting ...");
        bw.write("starting predicting ...\n");
        stopwatch.reset();
        stopwatch.start();
        double accuracy = Accuracy.accuracy(naiveBayes,dataSet);
        bw.write("Accuracy on training set: " + accuracy + "\n");
        bw.write("Prediction time: " + stopwatch.stop() + "\n");
        System.out.println("Accuracy on training set: " + accuracy);
        System.out.println("Prediction time: " + stopwatch.stop());

        System.out.println();
        bw.write("\nstarting predicting ...\n");
        System.out.println("starting predicting ...");
        stopwatch.reset();
        stopwatch.start();
        accuracy = Accuracy.accuracy(naiveBayes, testDataset);
        bw.write("Accuracy on testing set: " + accuracy + "\n");
        bw.write("Prediction time: " + stopwatch.stop() + "\n\n");
        System.out.println("Accuracy on testing set: " + accuracy);
        System.out.println("Prediction time: " + stopwatch.stop());
        System.out.println();
        bw.close();
    }

    private static void localBernoulliBabyTest() throws IOException, ClassNotFoundException, InvocationTargetException, NoSuchMethodException, InstantiationException, IllegalAccessException {
        BufferedWriter bw = new BufferedWriter(new FileWriter("~/baby_bernoulli.log"));
        System.out.println("------------Bernoulli Baby-------------");
        bw.write("------------Bernoulli Baby-------------\n");
        ClfDataSet dataSet = TRECFormat.loadClfDataSet(new File(DATASETS, "amazon_baby/11/train.trec"),
                DataSetType.CLF_SPARSE, true);
        ClfDataSet testDataset = TRECFormat.loadClfDataSet(new File(DATASETS, "amazon_baby/11/test.trec"),
                DataSetType.CLF_SPARSE, true);

        System.out.println(dataSet.getMetaInfo());
        System.out.println("Starting training ...");
        bw.write(dataSet.getMetaInfo()+"\n");
        bw.write("Starting training ...\n");

        Stopwatch stopwatch = Stopwatch.createStarted();
        NaiveBayes<Bernoulli> naiveBayes = new NaiveBayes<>(Bernoulli.class);
        naiveBayes.build(dataSet);
        System.out.println("Training time: " + stopwatch.stop());
        System.out.println("training done...");
        bw.write("Training time: " + stopwatch.stop() + "\n");
        bw.write("training done...\n");
        bw.write("\n");

        System.out.println();

        System.out.println("starting predicting ...");
        bw.write("starting predicting ...\n");
        stopwatch.reset();
        stopwatch.start();
        double accuracy = Accuracy.accuracy(naiveBayes,dataSet);
        bw.write("Accuracy on training set: " + accuracy + "\n");
        bw.write("Prediction time: " + stopwatch.stop() + "\n");
        System.out.println("Accuracy on training set: " + accuracy);
        System.out.println("Prediction time: " + stopwatch.stop());

        System.out.println();
        bw.write("\nstarting predicting ...\n");
        System.out.println("starting predicting ...");
        stopwatch.reset();
        stopwatch.start();
        accuracy = Accuracy.accuracy(naiveBayes, testDataset);
        bw.write("Accuracy on testing set: " + accuracy + "\n");
        bw.write("Prediction time: " + stopwatch.stop() + "\n\n");
        System.out.println("Accuracy on testing set: " + accuracy);
        System.out.println("Prediction time: " + stopwatch.stop());
        System.out.println();
        bw.close();
    }

    private static void localHistBabyTest(int maxBins) throws IOException, ClassNotFoundException, InvocationTargetException, NoSuchMethodException, InstantiationException, IllegalAccessException {
        BufferedWriter bw = new BufferedWriter(new FileWriter("~/baby_hist.log"));
        System.out.println("------------Hist Baby-------------");
        bw.write("------------Hist Baby-------------\n");
        ClfDataSet dataSet = TRECFormat.loadClfDataSet(new File(DATASETS, "amazon_baby/11/train.trec"),
                DataSetType.CLF_SPARSE, true);
        ClfDataSet testDataset = TRECFormat.loadClfDataSet(new File(DATASETS, "amazon_baby/11/test.trec"),
                DataSetType.CLF_SPARSE, true);

        System.out.println(dataSet.getMetaInfo());
        System.out.println("Starting training ...");
        bw.write(dataSet.getMetaInfo()+"\n");
        bw.write("Starting training ...\n");

        Stopwatch stopwatch = Stopwatch.createStarted();
        NaiveBayes<Histogram> naiveBayes = new NaiveBayes<>(Histogram.class);
        naiveBayes.build(dataSet, maxBins);
        System.out.println("Training time: " + stopwatch.stop());
        System.out.println("training done...");
        bw.write("Training time: " + stopwatch.stop() + "\n");
        bw.write("training done...\n");
        bw.write("\n");

        System.out.println();

        System.out.println("starting predicting ...");
        bw.write("starting predicting ...\n");
        stopwatch.reset();
        stopwatch.start();
        double accuracy = Accuracy.accuracy(naiveBayes,dataSet);
        bw.write("Accuracy on training set: " + accuracy + "\n");
        bw.write("Prediction time: " + stopwatch.stop() + "\n");
        System.out.println("Accuracy on training set: " + accuracy);
        System.out.println("Prediction time: " + stopwatch.stop());

        System.out.println();
        bw.write("\nstarting predicting ...\n");
        System.out.println("starting predicting ...");
        stopwatch.reset();
        stopwatch.start();
        accuracy = Accuracy.accuracy(naiveBayes, testDataset);
        bw.write("Accuracy on testing set: " + accuracy + "\n");
        bw.write("Prediction time: " + stopwatch.stop() + "\n\n");
        System.out.println("Accuracy on testing set: " + accuracy);
        System.out.println("Prediction time: " + stopwatch.stop());
        System.out.println();
        bw.close();
    }

    private static void localMulBabyTest() throws IOException, ClassNotFoundException, InvocationTargetException, NoSuchMethodException, InstantiationException, IllegalAccessException {
        BufferedWriter bw = new BufferedWriter(new FileWriter("~/baby_mul.log"));
        System.out.println("------------Multi Baby-------------");
        bw.write("------------Multi Baby-------------\n");
        ClfDataSet dataSet = TRECFormat.loadClfDataSet(new File(DATASETS, "amazon_baby/11/train.trec"),
                DataSetType.CLF_SPARSE, true);
        ClfDataSet testDataset = TRECFormat.loadClfDataSet(new File(DATASETS, "amazon_baby/11/test.trec"),
                DataSetType.CLF_SPARSE, true);

        System.out.println(dataSet.getMetaInfo());
        System.out.println("Starting training ...");
        bw.write(dataSet.getMetaInfo()+"\n");
        bw.write("Starting training ...\n");

        Stopwatch stopwatch = Stopwatch.createStarted();
        NaiveBayes<Multinomial> naiveBayes = new NaiveBayes<>(Multinomial.class);
        naiveBayes.build(dataSet);
        System.out.println("Training time: " + stopwatch.stop());
        System.out.println("training done...");
        bw.write("Training time: " + stopwatch.stop() + "\n");
        bw.write("training done...\n");
        bw.write("\n");

        System.out.println();

        System.out.println("starting predicting ...");
        bw.write("starting predicting ...\n");
        stopwatch.reset();
        stopwatch.start();
        double accuracy = Accuracy.accuracy(naiveBayes,dataSet);
        bw.write("Accuracy on training set: " + accuracy + "\n");
        bw.write("Prediction time: " + stopwatch.stop() + "\n");
        System.out.println("Accuracy on training set: " + accuracy);
        System.out.println("Prediction time: " + stopwatch.stop());

        System.out.println();
        bw.write("\nstarting predicting ...\n");
        System.out.println("starting predicting ...");
        stopwatch.reset();
        stopwatch.start();
        accuracy = Accuracy.accuracy(naiveBayes, testDataset);
        bw.write("Accuracy on testing set: " + accuracy + "\n");
        bw.write("Prediction time: " + stopwatch.stop() + "\n\n");
        System.out.println("Accuracy on testing set: " + accuracy);
        System.out.println("Prediction time: " + stopwatch.stop());
        System.out.println();
        bw.close();
    }

    private static void localGaussianBabyTest() throws IOException, ClassNotFoundException, InvocationTargetException, NoSuchMethodException, InstantiationException, IllegalAccessException {
        BufferedWriter bw = new BufferedWriter(new FileWriter("~/baby_gaussian.log"));
        System.out.println("------------Gaussian Baby-------------");
        bw.write("------------Gaussian Baby-------------\n");
        ClfDataSet dataSet = TRECFormat.loadClfDataSet(new File(DATASETS, "amazon_baby/11/train.trec"),
                DataSetType.CLF_SPARSE, true);
        ClfDataSet testDataset = TRECFormat.loadClfDataSet(new File(DATASETS, "amazon_baby/11/test.trec"),
                DataSetType.CLF_SPARSE, true);

        System.out.println(dataSet.getMetaInfo());
        System.out.println("Starting training ...");
        bw.write(dataSet.getMetaInfo()+"\n");
        bw.write("Starting training ...\n");

        Stopwatch stopwatch = Stopwatch.createStarted();
        NaiveBayes<Gaussian> naiveBayes = new NaiveBayes<>(Gaussian.class);
        naiveBayes.build(dataSet);
        System.out.println("Training time: " + stopwatch.stop());
        System.out.println("training done...");
        bw.write("Training time: " + stopwatch.stop() + "\n");
        bw.write("training done...\n");
        bw.write("\n");

        System.out.println();

        System.out.println("starting predicting ...");
        bw.write("starting predicting ...\n");
        stopwatch.reset();
        stopwatch.start();
        double accuracy = Accuracy.accuracy(naiveBayes,dataSet);
        bw.write("Accuracy on training set: " + accuracy + "\n");
        bw.write("Prediction time: " + stopwatch.stop() + "\n");
        System.out.println("Accuracy on training set: " + accuracy);
        System.out.println("Prediction time: " + stopwatch.stop());

        System.out.println();
        bw.write("\nstarting predicting ...\n");
        System.out.println("starting predicting ...");
        stopwatch.reset();
        stopwatch.start();
        accuracy = Accuracy.accuracy(naiveBayes, testDataset);
        bw.write("Accuracy on testing set: " + accuracy + "\n");
        bw.write("Prediction time: " + stopwatch.stop() + "\n\n");
        System.out.println("Accuracy on testing set: " + accuracy);
        System.out.println("Prediction time: " + stopwatch.stop());
        System.out.println();
        bw.close();
    }

}
