package edu.neu.ccs.pyramid.application;

import edu.neu.ccs.pyramid.configuration.Config;
import edu.neu.ccs.pyramid.dataset.DataSet;
import edu.neu.ccs.pyramid.dataset.DataSetUtil;
import edu.neu.ccs.pyramid.optimization.GradientDescent;
import edu.neu.ccs.pyramid.optimization.LBFGS;
import edu.neu.ccs.pyramid.optimization.Optimizer;
import edu.neu.ccs.pyramid.optimization.Terminator;
import edu.neu.ccs.pyramid.optimization.customize.SupervisedEmbeddingTSNELoss;
import org.apache.commons.lang3.time.StopWatch;
import org.apache.logging.log4j.Level;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.core.LoggerContext;
import org.apache.logging.log4j.core.config.Configuration;
import org.apache.logging.log4j.core.config.LoggerConfig;

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;

/**
 * Created by yuyuxu on 5/5/17.
 */
public class SupervisedEmbeddingTSNE {
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
        // X0, Y0, U, X, Y
        // precision, alpha, beta, gamma, omega
        // numEmbedding, sizeEmbedding
        // numIter, optimizer
        DataSet X0 = DataSetUtil.loadFeatureMatrixFromCSV(
                config.getString("X0"), config.getInt("numEmbedding"), config.getInt("sizeEmbedding"));
        DataSet Y0 = DataSetUtil.loadFeatureMatrixFromCSV(
                config.getString("Y0"), config.getInt("numEmbedding"), 2);
        DataSet U = DataSetUtil.loadFeatureMatrixFromCSV(
                config.getString("U"), config.getInt("numEmbedding"), 2);
        double[] precision = new double[config.getInt("numEmbedding")];
        try {
            BufferedReader br = new BufferedReader(new FileReader(config.getString("precision")));
            int i = 0;
            String line = br.readLine();
            while (line != null) {
                precision[i] = Double.parseDouble(line);
                i += 1;
                line = br.readLine();
            }
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        } catch (IOException e) {
            e.printStackTrace();
        }

        SupervisedEmbeddingTSNELoss function = new SupervisedEmbeddingTSNELoss(
                X0, Y0, U,
                precision, config.getDouble("alpha"), config.getDouble("beta"),
                config.getDouble("gamma"), config.getDouble("omega"));

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
        optimizer.getTerminator().setRelativeEpsilon(0.01);
        optimizer.getTerminator().setMaxStableIterations(10);
        optimizer.getTerminator().setMaxIteration(config.getInt("numIter"));
        System.out.println("==========================================\n");
        System.out.println("finished loading data and initializing ... \n");

        StopWatch stopWatch = new StopWatch();
        stopWatch.start();
        int i = 0;
        while (!optimizer.getTerminator().shouldTerminate()) {
            optimizer.iterate();
            System.out.println("iter=" + (i++) + ", loss=" + function.getValue() + ", time=" + stopWatch);
        }
        System.out.println("==========================================\n");
        System.out.println("finished updating ... \n");

        DataSetUtil.saveFeatureMatrixToCSV(config.getString("X"), function.getX());
        DataSetUtil.saveFeatureMatrixToCSV(config.getString("Y"), function.getY());
    }
}
