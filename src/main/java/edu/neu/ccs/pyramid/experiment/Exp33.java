package edu.neu.ccs.pyramid.experiment;

import edu.neu.ccs.pyramid.classification.logistic_regression.LogisticLoss;
import edu.neu.ccs.pyramid.classification.logistic_regression.LogisticRegression;
import edu.neu.ccs.pyramid.configuration.Config;
import edu.neu.ccs.pyramid.dataset.ClfDataSet;
import edu.neu.ccs.pyramid.dataset.DataSetType;
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

        train(config);
    }

    private static void train(Config config) throws Exception{
        String input = config.getString("input.folder");
        ClfDataSet dataSet = TRECFormat.loadClfDataSet(new File(input,"train.trec"),
                DataSetType.CLF_SPARSE, true);
        ClfDataSet testSet = TRECFormat.loadClfDataSet(new File(input,"test.trec"),
                DataSetType.CLF_SPARSE, true);
        System.out.println(dataSet.getMetaInfo());

        LogisticRegression logisticRegression = new LogisticRegression(dataSet.getNumClasses(),dataSet.getNumFeatures());
        logisticRegression.setFeatureExtraction(false);
        LogisticLoss function = new LogisticLoss(logisticRegression,dataSet,config.getDouble("gaussianPriorVariance"));
        LBFGS lbfgs = new LBFGS(function);
        lbfgs.setHistory(config.getInt("historyLength"));
        for (int i=0;i<config.getInt("numIterations");i++){
            System.out.println("--------");
            System.out.println("iteration "+i);
            lbfgs.iterate();
            System.out.println("loss: " + function.getValue(logisticRegression.getWeights().getAllWeights()));
            System.out.println("train: "+ Accuracy.accuracy(logisticRegression, dataSet));
            System.out.println("test: "+Accuracy.accuracy(logisticRegression,testSet));
        }
    }

}
