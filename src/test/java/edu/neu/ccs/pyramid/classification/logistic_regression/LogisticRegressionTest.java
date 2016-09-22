package edu.neu.ccs.pyramid.classification.logistic_regression;

import edu.neu.ccs.pyramid.configuration.Config;
import edu.neu.ccs.pyramid.dataset.ClfDataSet;
import edu.neu.ccs.pyramid.dataset.DataSetType;
import edu.neu.ccs.pyramid.dataset.TRECFormat;
import edu.neu.ccs.pyramid.eval.Accuracy;
import edu.neu.ccs.pyramid.optimization.ConjugateGradientDescent;
import edu.neu.ccs.pyramid.optimization.GradientDescent;
import edu.neu.ccs.pyramid.optimization.LBFGS;
import org.apache.logging.log4j.Level;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.core.LoggerContext;
import org.apache.logging.log4j.core.config.Configuration;
import org.apache.logging.log4j.core.config.LoggerConfig;

import java.io.File;

public class LogisticRegressionTest {
    private static final Config config = new Config("config/local.properties");
    private static final String DATASETS = config.getString("input.datasets");
    private static final String TMP = config.getString("output.tmp");

    public static void main(String[] args) throws Exception{
        LoggerContext ctx = (LoggerContext) LogManager.getContext(false);
        Configuration config = ctx.getConfiguration();
        LoggerConfig loggerConfig = config.getLoggerConfig(LogManager.ROOT_LOGGER_NAME);
        loggerConfig.setLevel(Level.DEBUG);
        ctx.updateLoggers();
        test3();

    }

    private static void test1() throws Exception{

        ClfDataSet dataSet = TRECFormat.loadClfDataSet(new File(DATASETS, "/imdb/3/train.trec"),
                DataSetType.CLF_SPARSE, false);
        ClfDataSet testSet = TRECFormat.loadClfDataSet(new File(DATASETS, "/imdb/3/test.trec"),
                DataSetType.CLF_SPARSE, false);
        System.out.println(dataSet.getMetaInfo());

        LogisticRegression logisticRegression = new LogisticRegression(dataSet.getNumClasses(),dataSet.getNumFeatures());
        LogisticLoss function = new LogisticLoss(logisticRegression,dataSet,1000, true);
        GradientDescent gradientDescent = new GradientDescent(function);
        gradientDescent.getLineSearcher().setInitialStepLength(1.0E-4);
        gradientDescent.optimize();
        System.out.println("train: "+Accuracy.accuracy(logisticRegression,dataSet));
        System.out.println("test: "+Accuracy.accuracy(logisticRegression,testSet));

    }


    private static void test2() throws Exception{

        ClfDataSet dataSet = TRECFormat.loadClfDataSet(new File(DATASETS, "/imdb/3/train.trec"),
                DataSetType.CLF_SPARSE, true);
        ClfDataSet testSet = TRECFormat.loadClfDataSet(new File(DATASETS, "/imdb/3/test.trec"),
                DataSetType.CLF_SPARSE, true);
        System.out.println(dataSet.getMetaInfo());

        LogisticRegression logisticRegression = new LogisticRegression(dataSet.getNumClasses(),dataSet.getNumFeatures());
        LogisticLoss function = new LogisticLoss(logisticRegression,dataSet,0.1, true);
        ConjugateGradientDescent conjugateGradientDescent = new ConjugateGradientDescent(function);
        conjugateGradientDescent.getLineSearcher().setInitialStepLength(0.01);
        conjugateGradientDescent.optimize();
        System.out.println("train: "+Accuracy.accuracy(logisticRegression,dataSet));
        System.out.println("test: "+Accuracy.accuracy(logisticRegression,testSet));

    }




    private static void test3() throws Exception{

        ClfDataSet dataSet = TRECFormat.loadClfDataSet(new File(DATASETS, "/imdb/3/train.trec"),
                DataSetType.CLF_SPARSE, false);
        ClfDataSet testSet = TRECFormat.loadClfDataSet(new File(DATASETS, "/imdb/3/test.trec"),
                DataSetType.CLF_SPARSE, false);
        System.out.println(dataSet.getMetaInfo());

        LogisticRegression logisticRegression = new LogisticRegression(dataSet.getNumClasses(),dataSet.getNumFeatures());
        LogisticLoss function = new LogisticLoss(logisticRegression,dataSet,0.1, true);
        LBFGS lbfgs = new LBFGS(function);
        lbfgs.optimize();
        System.out.println("train: "+Accuracy.accuracy(logisticRegression,dataSet));
        System.out.println("test: "+Accuracy.accuracy(logisticRegression,testSet));

    }

}