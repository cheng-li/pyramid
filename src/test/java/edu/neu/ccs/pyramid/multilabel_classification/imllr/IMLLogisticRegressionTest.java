package edu.neu.ccs.pyramid.multilabel_classification.imllr;

import edu.neu.ccs.pyramid.configuration.Config;
import edu.neu.ccs.pyramid.dataset.*;
import edu.neu.ccs.pyramid.eval.Accuracy;
import edu.neu.ccs.pyramid.eval.Overlap;

import org.apache.commons.lang3.time.StopWatch;
import org.apache.logging.log4j.Level;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.core.LoggerContext;
import org.apache.logging.log4j.core.config.Configuration;
import org.apache.logging.log4j.core.config.LoggerConfig;

import java.io.File;
import java.util.List;

public class IMLLogisticRegressionTest {
    private static final Config config = new Config("configs/local.config");
    private static final String DATASETS = config.getString("input.datasets");
    private static final String TMP = config.getString("output.tmp");

    public static void main(String[] args) throws Exception{
        LoggerContext ctx = (LoggerContext) LogManager.getContext(false);
        Configuration config = ctx.getConfiguration();
        LoggerConfig loggerConfig = config.getLoggerConfig(LogManager.ROOT_LOGGER_NAME);
        loggerConfig.setLevel(Level.DEBUG);
        ctx.updateLoggers();
        test1();
    }

    static void test1() throws Exception{
        MultiLabelClfDataSet dataSet = TRECFormat.loadMultiLabelClfDataSet(new File(DATASETS, "ohsumed/3/train.trec"), DataSetType.ML_CLF_SPARSE, true);
        MultiLabelClfDataSet testSet = TRECFormat.loadMultiLabelClfDataSet(new File(DATASETS, "ohsumed/3/test.trec"), DataSetType.ML_CLF_SPARSE, true);
        List<MultiLabel> assignments = DataSetUtil.gatherMultiLabels(dataSet);
        IMLLogisticTrainer trainer = IMLLogisticTrainer.getBuilder().setEpsilon(0.01).setGaussianPriorVariance(1)
                .setHistory(5).build();
        StopWatch stopWatch = new StopWatch();
        stopWatch.start();
        IMLLogisticRegression logisticRegression =trainer.train(dataSet,assignments);
        System.out.println(stopWatch);


        System.out.println("training accuracy="+ Accuracy.accuracy(logisticRegression, dataSet));
        System.out.println("training overlap = "+ Overlap.overlap(logisticRegression, dataSet));
        System.out.println("test accuracy="+ Accuracy.accuracy(logisticRegression, testSet));
        System.out.println("test overlap = "+ Overlap.overlap(logisticRegression,testSet));
    }

}