package edu.neu.ccs.pyramid.experiment;

import edu.neu.ccs.pyramid.classification.logistic_regression.LogisticLoss;
import edu.neu.ccs.pyramid.classification.logistic_regression.LogisticRegression;
import edu.neu.ccs.pyramid.classification.logistic_regression.LogisticRegressonInspector;
import edu.neu.ccs.pyramid.classification.logistic_regression.RidgeLogisticTrainer;
import edu.neu.ccs.pyramid.configuration.Config;
import edu.neu.ccs.pyramid.dataset.ClfDataSet;
import edu.neu.ccs.pyramid.dataset.DataSetType;
import edu.neu.ccs.pyramid.dataset.LabelTranslator;
import edu.neu.ccs.pyramid.dataset.TRECFormat;
import edu.neu.ccs.pyramid.eval.Accuracy;
import edu.neu.ccs.pyramid.optimization.LBFGS;

import java.io.File;

/**
 * logistic regression
 * Created by chengli on 12/12/14.
 */
public class Exp33 {
    public static void main(String[] args) throws Exception{
        if (args.length !=1){
            throw new IllegalArgumentException("please specify the config file");
        }

        Config config = new Config(args[0]);
        System.out.println(config);

        if (config.getBoolean("train")){
            train(config);
        }

        if (config.getBoolean("verify")){
            verify(config);
        }

    }

    public static void mainFromConfig(Config config) throws Exception{
        if (config.getBoolean("train")){
            train(config);
        }

        if (config.getBoolean("verify")){
            verify(config);
        }

    }

    private static void train(Config config) throws Exception{
        String input = config.getString("input.folder");
        ClfDataSet dataSet = TRECFormat.loadClfDataSet(new File(input,"train.trec"),
                DataSetType.CLF_SPARSE, true);
        ClfDataSet testSet = TRECFormat.loadClfDataSet(new File(input,"test.trec"),
                DataSetType.CLF_SPARSE, true);
        System.out.println(dataSet.getMetaInfo());

        RidgeLogisticTrainer trainer = RidgeLogisticTrainer.getBuilder()
                .setHistory(config.getInt("history"))
                .setGaussianPriorVariance(config.getDouble("gaussianPriorVariance"))
                .setEpsilon(config.getDouble("epsilon"))
                .build();



        LogisticRegression logisticRegression = trainer.train(dataSet);
        System.out.println("train: "+ Accuracy.accuracy(logisticRegression, dataSet));
        System.out.println("test: "+Accuracy.accuracy(logisticRegression,testSet));
        File modelFile = new File(config.getString("archive.folder"),config.getString("archive.model"));
        logisticRegression.serialize(modelFile);

    }

    private static void verify(Config config) throws Exception{
        String input = config.getString("input.folder");
        ClfDataSet dataSet = TRECFormat.loadClfDataSet(new File(input,"train.trec"),
                DataSetType.CLF_SPARSE, true);
        LabelTranslator labelTranslator = dataSet.getSetting().getLabelTranslator();
        File modelFile = new File(config.getString("archive.folder"),config.getString("archive.model"));
        LogisticRegression logisticRegression = LogisticRegression.deserialize(modelFile);
        int limit = config.getInt("verify.topFeature.limit");
        for (int k=0;k<logisticRegression.getNumClasses();k++){
            System.out.println("top feature for class "+k+"("+labelTranslator.toExtLabel(k)+")");
            System.out.println(LogisticRegressonInspector.topFeatures(logisticRegression,k,limit));
        }


    }

}
