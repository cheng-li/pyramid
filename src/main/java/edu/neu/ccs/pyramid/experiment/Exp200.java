package edu.neu.ccs.pyramid.experiment;

import com.google.common.base.Stopwatch;
import edu.neu.ccs.pyramid.classification.naive_bayes.*;
import edu.neu.ccs.pyramid.configuration.Config;
import edu.neu.ccs.pyramid.dataset.ClfDataSet;
import edu.neu.ccs.pyramid.dataset.DataSetType;
import edu.neu.ccs.pyramid.dataset.TRECFormat;
import edu.neu.ccs.pyramid.eval.Accuracy;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.lang.reflect.InvocationTargetException;

/**
 * Created by Rainicy on 5/18/15.
 */
public class Exp200 {

    public static void main(String[] args) throws IOException, ClassNotFoundException, InvocationTargetException, NoSuchMethodException, InstantiationException, IllegalAccessException {
        if (args.length !=1){
            throw new IllegalArgumentException("please specify the config file");
        }
        System.out.println(args[0]);
        Config config = new Config(args[0]);
        System.out.println(config);

        String DATASETS = config.getString("input.datasets");
        String train = config.getString("train");
        String test = config.getString("test");
        int numBins = config.getInt("numBins");
        String topic = config.getString("topic");
        String logs = config.getString("output.log");

        ClfDataSet dataSet = TRECFormat.loadClfDataSet(new File(DATASETS, train),
                DataSetType.CLF_SPARSE, true);
        ClfDataSet testDataset = TRECFormat.loadClfDataSet(new File(DATASETS, test),
                DataSetType.CLF_SPARSE, true);

        BufferedWriter bw = new BufferedWriter(new FileWriter(logs));
        System.out.println(dataSet.getMetaInfo());
        bw.write(dataSet.getMetaInfo()+"\n");

        System.out.println("------------"+ topic +"-------------");
        bw.write("------------" + topic + "-------------\n");


        // 1) Bernoulli
        System.out.println("============Bernoulli============");
        bw.write("============Bernoulli============\n");
        System.out.println("Starting training ...");
        bw.write("Starting training ...\n");

        Stopwatch stopwatch = Stopwatch.createStarted();
        NaiveBayes<Bernoulli> naiveBayes = new NaiveBayes<>(Bernoulli.class);
        naiveBayes.build(dataSet);
        System.out.println("training done...");
        bw.write("Training time: " + stopwatch.stop() + "\n");
        bw.write("training done...\n");
        bw.write("\n");

        System.out.println();

        System.out.println("starting predicting ...");
        bw.write("starting predicting ...\n");
        stopwatch.reset();
        stopwatch.start();
        double accuracy = Accuracy.accuracy(naiveBayes, dataSet);
        bw.write("Accuracy on training set: " + accuracy + "\n");
        bw.write("Prediction time: " + stopwatch.stop() + "\n");
        System.out.println("Accuracy on training set: " + accuracy);

        System.out.println();
        bw.write("\nstarting predicting ...\n");
        System.out.println("starting predicting ...");
        stopwatch.reset();
        stopwatch.start();
        accuracy = Accuracy.accuracy(naiveBayes, testDataset);
        bw.write("Accuracy on testing set: " + accuracy + "\n");
        bw.write("Prediction time: " + stopwatch.stop() + "\n\n");
        System.out.println("Accuracy on testing set: " + accuracy);
        System.out.println();


        // 2) Histogram
        System.out.println("============Histogram============");
        bw.write("============Histogram============\n");
        System.out.println("Starting training ...");
        bw.write("Starting training ...\n");

        stopwatch.reset();
        stopwatch.start();
        NaiveBayes<Histogram> naiveBayes1 = new NaiveBayes<>(Histogram.class);
        naiveBayes1.build(dataSet, numBins);
        System.out.println("training done...");
        bw.write("Training time: " + stopwatch.stop() + "\n");
        bw.write("training done...\n");
        bw.write("\n");

        System.out.println();

        System.out.println("starting predicting ...");
        bw.write("starting predicting ...\n");
        stopwatch.reset();
        stopwatch.start();
        accuracy = Accuracy.accuracy(naiveBayes1, dataSet);
        bw.write("Accuracy on training set: " + accuracy + "\n");
        bw.write("Prediction time: " + stopwatch.stop() + "\n");
        System.out.println("Accuracy on training set: " + accuracy);

        System.out.println();
        bw.write("\nstarting predicting ...\n");
        System.out.println("starting predicting ...");
        stopwatch.reset();
        stopwatch.start();
        accuracy = Accuracy.accuracy(naiveBayes1, testDataset);
        bw.write("Accuracy on testing set: " + accuracy + "\n");
        bw.write("Prediction time: " + stopwatch.stop() + "\n\n");
        System.out.println("Accuracy on testing set: " + accuracy);
        System.out.println();


        // 3) Multinomial
        System.out.println("============Multinomial============");
        bw.write("============Multinomial============\n");
        System.out.println("Starting training ...");
        bw.write("Starting training ...\n");

        stopwatch.reset();
        stopwatch.start();
        NaiveBayes<Multinomial> naiveBayes2 = new NaiveBayes<>(Multinomial.class);
        naiveBayes2.build(dataSet);
        System.out.println("training done...");
        bw.write("Training time: " + stopwatch.stop() + "\n");
        bw.write("training done...\n");
        bw.write("\n");

        System.out.println();

        System.out.println("starting predicting ...");
        bw.write("starting predicting ...\n");
        stopwatch.reset();
        stopwatch.start();
        accuracy = Accuracy.accuracy(naiveBayes2, dataSet);
        bw.write("Accuracy on training set: " + accuracy + "\n");
        bw.write("Prediction time: " + stopwatch.stop() + "\n");
        System.out.println("Accuracy on training set: " + accuracy);

        System.out.println();
        bw.write("\nstarting predicting ...\n");
        System.out.println("starting predicting ...");
        stopwatch.reset();
        stopwatch.start();
        accuracy = Accuracy.accuracy(naiveBayes2, testDataset);
        bw.write("Accuracy on testing set: " + accuracy + "\n");
        bw.write("Prediction time: " + stopwatch.stop() + "\n\n");
        System.out.println("Accuracy on testing set: " + accuracy);
        System.out.println();

        // 4) Gaussian
        System.out.println("============Gaussian============");
        bw.write("============Gaussian============\n");
        System.out.println("Starting training ...");
        bw.write("Starting training ...\n");

        stopwatch.reset();
        stopwatch.start();
        NaiveBayes<Gaussian> naiveBayes3 = new NaiveBayes<>(Gaussian.class);
        naiveBayes3.build(dataSet);
        System.out.println("training done...");
        bw.write("Training time: " + stopwatch.stop() + "\n");
        bw.write("training done...\n");
        bw.write("\n");

        System.out.println();

        System.out.println("starting predicting ...");
        bw.write("starting predicting ...\n");
        stopwatch.reset();
        stopwatch.start();
        accuracy = Accuracy.accuracy(naiveBayes3, dataSet);
        bw.write("Accuracy on training set: " + accuracy + "\n");
        bw.write("Prediction time: " + stopwatch.stop() + "\n");
        System.out.println("Accuracy on training set: " + accuracy);

        System.out.println();
        bw.write("\nstarting predicting ...\n");
        System.out.println("starting predicting ...");
        stopwatch.reset();
        stopwatch.start();
        accuracy = Accuracy.accuracy(naiveBayes3, testDataset);
        bw.write("Accuracy on testing set: " + accuracy + "\n");
        bw.write("Prediction time: " + stopwatch.stop() + "\n\n");
        System.out.println("Accuracy on testing set: " + accuracy);
        System.out.println();

        bw.close();
    }
}
