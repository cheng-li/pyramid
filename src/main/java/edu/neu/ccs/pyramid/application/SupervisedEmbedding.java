package edu.neu.ccs.pyramid.application;

import edu.neu.ccs.pyramid.configuration.Config;
import edu.neu.ccs.pyramid.dataset.DataSet;
import edu.neu.ccs.pyramid.dataset.DataSetUtil;
import edu.neu.ccs.pyramid.optimization.GradientDescent;
import edu.neu.ccs.pyramid.optimization.LBFGS;
import edu.neu.ccs.pyramid.optimization.Optimizer;
import edu.neu.ccs.pyramid.optimization.Terminator;
import edu.neu.ccs.pyramid.optimization.customize.SupervisedEmbeddingLoss;
import org.apache.commons.lang3.time.StopWatch;
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
        //   inputEmbedding
        //   inputProjection
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
        DataSet embedding = DataSetUtil.loadFeatureMatrixFromCSV(
                config.getString("inputEmbedding"), config.getInt("numEmbedding"), config.getInt("sizeEmbedding"));
        System.out.println(embedding.getMetaInfo());
        DataSet projection = DataSetUtil.loadFeatureMatrixFromCSV(
                config.getString("inputProjection"), config.getInt("numEmbedding"), 2);
        System.out.println(projection.getMetaInfo());
        System.out.println("==========================================\n");

        SupervisedEmbeddingLoss function = new SupervisedEmbeddingLoss(
                transform, embedding, projection, config.getDouble("alpha"), config.getDouble("beta"));

        Optimizer.Iterative optimizer = null;
        if (config.getString("optimizer").equals("gd")) {
            optimizer = new GradientDescent(function);
        } else if (config.getString("optimizer").equals("lbfgs")) {
            optimizer = new LBFGS(function);
        } else {
            System.out.println("Error: optimizer " + config.getString("optimizer") + " not supported!");
            System.exit(0);
        }
        optimizer.getTerminator().setOperation(Terminator.Operation.OR);
        optimizer.getTerminator().setAbsoluteEpsilon(0.001);
        optimizer.getTerminator().setRelativeEpsilon(0.1);
        optimizer.getTerminator().setMaxStableIterations(5);
        optimizer.getTerminator().setMaxIteration(config.getInt("numIter"));

        StopWatch stopWatch = new StopWatch();
        stopWatch.start();
        int i = 0;
        while (!optimizer.getTerminator().shouldTerminate()) {
            optimizer.iterate();
            System.out.println("iter=" + (i++) + ", loss=" + function.getValue() + ", time=" + stopWatch);
        }
        System.out.println("==========================================\n");
        System.out.println("finished updating ... \n");

        DataSet finalEmbedding = function.getUpdatedEmbedding();
        DataSetUtil.saveFeatureMatrixToCSV(config.getString("outputEmbedding"), finalEmbedding);
        DataSet finalProjection = function.getUpdatedProjection();
        DataSetUtil.saveFeatureMatrixToCSV(config.getString("outputProjection"), finalProjection);
    }
}
