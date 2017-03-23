package edu.neu.ccs.pyramid.application;

import edu.neu.ccs.pyramid.configuration.Config;
import edu.neu.ccs.pyramid.dataset.DataSet;
import edu.neu.ccs.pyramid.dataset.DataSetUtil;
import edu.neu.ccs.pyramid.optimization.GradientDescent;
import edu.neu.ccs.pyramid.optimization.customize.SupervisedEmbeddingLoss;
import org.apache.logging.log4j.Level;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.core.LoggerContext;
import org.apache.logging.log4j.core.config.Configuration;
import org.apache.logging.log4j.core.config.LoggerConfig;

/**
 * Created by yuyuxu on 3/23/17.
 */
public class SupervisedEmbedding {
    public static void main(String[] args) throws Exception {
        if (args.length != 1) {
            throw new IllegalArgumentException("Please specify a properties file.");
        }

        Config config = new Config(args[0]);
        main(config);
    }

    public static void main(Config config) throws Exception {
        LoggerContext ctx = (LoggerContext) LogManager.getContext(false);
        Configuration ctxConfig = ctx.getConfiguration();
        LoggerConfig loggerConfig = ctxConfig.getLoggerConfig(LogManager.ROOT_LOGGER_NAME);
        loggerConfig.setLevel(Level.INFO);
        ctx.updateLoggers();

        // All possible config:
        //   inputTransform
        //   inputDistance
        //   inputEmbedding
        //   numEmbedding
        //   sizeEmbedding
        //   outputEmbedding
        //   outputProjection
        //   alpha
        //   beta
        //   numIter
        DataSet transform = DataSetUtil.loadFeatureMatrixFromCSV(
                config.getString("inputTransform"), config.getInt("sizeEmbedding"), 2);
        System.out.println(transform.getMetaInfo());
        DataSet distance = DataSetUtil.loadFeatureMatrixFromCSV(
                config.getString("inputDistance"), config.getInt("numEmbedding"), config.getInt("numEmbedding"));
        System.out.println(distance.getMetaInfo());
        DataSet embedding = DataSetUtil.loadFeatureMatrixFromCSV(
                config.getString("inputEmbedding"), config.getInt("numEmbedding"), config.getInt("sizeEmbedding"));
        System.out.println(embedding.getMetaInfo());
        System.out.println("==========================================\n");

        SupervisedEmbeddingLoss function = new SupervisedEmbeddingLoss(
                distance, transform, embedding, config.getDouble("alpha"), config.getDouble("beta"));
        GradientDescent gd = new GradientDescent(function);
        for (int i = 0; i < config.getInt("numIter"); i++) {
            gd.iterate();
            System.out.println("loss=" + function.getValue());
        }
        System.out.println("==========================================\n");

        DataSet finalEmbedding = function.getUpdatedEmbedding();
        DataSetUtil.saveFeatureMatrixToCSV(config.getString("outputEmbedding"), finalEmbedding);
        DataSet finalProjection = function.getUpdatedProjection();
        DataSetUtil.saveFeatureMatrixToCSV(config.getString("outputProjection"), finalProjection);
    }
}
