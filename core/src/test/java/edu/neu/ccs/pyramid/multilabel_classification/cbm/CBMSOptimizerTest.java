package edu.neu.ccs.pyramid.multilabel_classification.cbm;

import edu.neu.ccs.pyramid.configuration.Config;
import edu.neu.ccs.pyramid.dataset.*;
import edu.neu.ccs.pyramid.eval.MLMeasures;
import junit.framework.TestCase;
import org.apache.logging.log4j.Level;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.core.LoggerContext;
import org.apache.logging.log4j.core.config.Configuration;
import org.apache.logging.log4j.core.config.LoggerConfig;

import java.io.File;
import java.util.Arrays;
import java.util.Set;
import java.util.stream.Collectors;

/**
 * Created by chengli on 11/17/16.
 */
public class CBMSOptimizerTest  {
    private static final Config config = new Config("config/local.properties");
    private static final String DATASETS = config.getString("input.datasets");
    private static final String TMP = config.getString("output.tmp");


    public static void main(String[] args) throws Exception{
//        LoggerContext ctx = (LoggerContext) LogManager.getContext(false);
//        Configuration config = ctx.getConfiguration();
//        LoggerConfig loggerConfig = config.getLoggerConfig(LogManager.ROOT_LOGGER_NAME);
//        loggerConfig.setLevel(Level.DEBUG);
//        ctx.updateLoggers();
        test1();
    }

    private static void test1() throws Exception{
        MultiLabelClfDataSet dataSet = TRECFormat.loadMultiLabelClfDataSet(new File(DATASETS, "scene/train"),
                DataSetType.ML_CLF_DENSE, true);
        MultiLabelClfDataSet testSet = TRECFormat.loadMultiLabelClfDataSet(new File(DATASETS, "scene/test"),
                DataSetType.ML_CLF_DENSE, true);
        int numComponents = 50;
        CBMS cbms = new CBMS(dataSet.getNumClasses(), numComponents, dataSet.getNumFeatures());
        Set<MultiLabel> seen = DataSetUtil.gatherMultiLabels(dataSet).stream().collect(Collectors.toSet());
        MultiLabel empty = new MultiLabel();
        if (seen.contains(empty)){
            cbms.setAllowEmpty(true);
            System.out.println("training set contains empty labels, automatically set predict.allowEmpty = true");
        } else {
            cbms.setAllowEmpty(false);
            System.out.println("training set does not contain empty labels, automatically set predict.allowEmpty = false");
        }

        CBMSOptimizer optimizer = new CBMSOptimizer(cbms, dataSet);
        optimizer.setPriorVarianceBinary(1);
        optimizer.setPriorVarianceMultiClass(1);
        optimizer.setComponentWeightsVariance(10);
        CBMSInitializer.initialize(cbms, dataSet,optimizer);
        System.out.println("after initialization");
        System.out.println("training performance");
        System.out.println(new MLMeasures(cbms, dataSet));
        System.out.println("test performance");
        System.out.println(new MLMeasures(cbms, testSet));

        for (int i=1;i<=50;i++){
            System.out.println("iteration "+i);
            optimizer.eStep();
            System.out.println("after E");
            System.out.println("objective = "+optimizer.getObjective());
            System.out.println("multi-class obj = "+optimizer.multiClassClassifierObj());
            System.out.println("binary obj = "+optimizer.binaryObj());
            optimizer.mStep();
            System.out.println("after M");
            System.out.println("objective = "+optimizer.getObjective());
            System.out.println("multi-class obj = "+optimizer.multiClassClassifierObj());
            System.out.println("binary obj = "+optimizer.binaryObj());
//            System.out.println(Arrays.toString(optimizer.gammas[0]));
//            System.out.println(Arrays.toString(cbms.predictClassProbs(dataSet.getRow(0))));
//            System.out.println(cbms.getBinaryClassifiers()[0]);
//            System.out.println(cbms.predictByMarginals(dataSet.getRow(0)));
//            System.out.println(cbms.predict(dataSet.getRow(199)));
            System.out.println("training performance");
            System.out.println(new MLMeasures(cbms, dataSet));
            System.out.println("test performance");
            System.out.println(new MLMeasures(cbms, testSet));
        }
    }

}