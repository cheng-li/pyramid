package edu.neu.ccs.pyramid.multilabel_classification.cbm;

import edu.neu.ccs.pyramid.configuration.Config;
import edu.neu.ccs.pyramid.dataset.DataSetType;
import edu.neu.ccs.pyramid.dataset.MultiLabelClfDataSet;
import edu.neu.ccs.pyramid.dataset.TRECFormat;
import edu.neu.ccs.pyramid.optimization.LBFGS;
import junit.framework.TestCase;
import org.apache.logging.log4j.Level;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.core.LoggerContext;
import org.apache.logging.log4j.core.config.Configuration;
import org.apache.logging.log4j.core.config.LoggerConfig;

import java.io.File;

/**
 * Created by chengli on 3/2/17.
 */
public class AugmentedLRLossTest {
    private static final Config config = new Config("config/local.properties");
    private static final String DATASETS = config.getString("input.datasets");
    private static final String TMP = config.getString("output.tmp");
    public static void main(String[] args) throws Exception{

        LoggerContext ctx = (LoggerContext) LogManager.getContext(false);
        Configuration config = ctx.getConfiguration();
        LoggerConfig loggerConfig = config.getLoggerConfig(LogManager.ROOT_LOGGER_NAME);
        loggerConfig.setLevel(Level.DEBUG);
        ctx.updateLoggers();

        MultiLabelClfDataSet dataSet = TRECFormat.loadMultiLabelClfDataSet(new File(DATASETS, "scene/train"),
                DataSetType.ML_CLF_DENSE, true);
        MultiLabelClfDataSet testSet = TRECFormat.loadMultiLabelClfDataSet(new File(DATASETS, "scene/test"),
                DataSetType.ML_CLF_DENSE, true);
        AugmentedLR augmentedLR = new AugmentedLR(dataSet.getNumFeatures(), 1);
        double[][] gammas = new double[dataSet.getNumDataPoints()][1];
        for (int i=0;i<dataSet.getNumDataPoints();i++){
            gammas[i][0]=1;
        }
        AugmentedLRLoss loss = new AugmentedLRLoss(dataSet, 0, gammas, augmentedLR, 1, false);
        LBFGS lbfgs = new LBFGS(loss);

        for (int i=0;i<100;i++){
            lbfgs.iterate();
            System.out.println(loss.getValue());
        }

    }

}