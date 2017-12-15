package edu.neu.ccs.pyramid.application;

import edu.neu.ccs.pyramid.configuration.Config;
import edu.neu.ccs.pyramid.dataset.DataSetType;
import edu.neu.ccs.pyramid.dataset.RegDataSet;
import edu.neu.ccs.pyramid.dataset.TRECFormat;
import edu.neu.ccs.pyramid.eval.RMSE;
import edu.neu.ccs.pyramid.regression.least_squares_boost.LSBoost;
import edu.neu.ccs.pyramid.regression.least_squares_boost.LSBoostOptimizer;
import edu.neu.ccs.pyramid.regression.least_squares_boost.PMMLConverter;
import edu.neu.ccs.pyramid.regression.regression_tree.RegTreeConfig;
import edu.neu.ccs.pyramid.regression.regression_tree.RegTreeFactory;
import edu.neu.ccs.pyramid.util.PrintUtil;
import edu.neu.ccs.pyramid.util.Serialization;
import org.apache.commons.io.FileUtils;

import java.io.File;
import java.io.IOException;
import java.util.logging.FileHandler;
import java.util.logging.Logger;
import java.util.logging.SimpleFormatter;

/**
 * least square gradient boosting for regression
 * Created by chengli on 9/22/16.
 */
public class GBRegressor {
    public static void main(String[] args) throws Exception{
        if (args.length !=1){
            throw new IllegalArgumentException("Please specify a properties file.");
        }
        Config config = new Config(args[0]);

        Logger logger = Logger.getAnonymousLogger();
        String logFile = config.getString("output.log");
        FileHandler fileHandler = null;
        if (!logFile.isEmpty()){
            new File(logFile).getParentFile().mkdirs();
            //todo should append?
            fileHandler = new FileHandler(logFile, true);
            java.util.logging.Formatter formatter = new SimpleFormatter();
            fileHandler.setFormatter(formatter);
            logger.addHandler(fileHandler);
            logger.setUseParentHandlers(false);
        }

        logger.info(config.toString());

        if (config.getBoolean("train")){
            train(config,logger);
        }

        if (config.getBoolean("test")){
            test(config,logger);
        }

        if (fileHandler!=null){
            fileHandler.close();
        }
    }

    public static void main(Config config) throws Exception{

        Logger logger = Logger.getAnonymousLogger();
        String logFile = config.getString("output.log");
        FileHandler fileHandler = null;
        if (!logFile.isEmpty()){
            new File(logFile).getParentFile().mkdirs();
            //todo should append?
            fileHandler = new FileHandler(logFile, true);
            java.util.logging.Formatter formatter = new SimpleFormatter();
            fileHandler.setFormatter(formatter);
            logger.addHandler(fileHandler);
            logger.setUseParentHandlers(false);
        }

        logger.info(config.toString());

        if (config.getBoolean("train")){
            train(config,logger);
        }

        if (config.getBoolean("test")){
            test(config,logger);
        }

        if (fileHandler!=null){
            fileHandler.close();
        }
    }

    private static void train(Config config, Logger logger) throws Exception{
        String sparsity = config.getString("input.matrixType");
        DataSetType dataSetType = null;
        switch (sparsity){
            case "dense":
                dataSetType = DataSetType.REG_DENSE;
                break;
            case "sparse":
                dataSetType = DataSetType.REG_SPARSE;
                break;
            default:
                throw new IllegalArgumentException("input.matrixType should be dense or sparse");
        }
        RegDataSet trainSet = TRECFormat.loadRegDataSet(config.getString("input.trainData"),dataSetType, true);

        RegDataSet testSet = null;
        if (config.getBoolean("train.showTestProgress")){
            testSet = TRECFormat.loadRegDataSet(config.getString("input.testData"),dataSetType, true);
        }

        LSBoost lsBoost = new LSBoost();

        RegTreeConfig regTreeConfig = new RegTreeConfig().setMaxNumLeaves(config.getInt("train.numLeaves"));
        RegTreeFactory regTreeFactory = new RegTreeFactory(regTreeConfig);
        LSBoostOptimizer optimizer = new LSBoostOptimizer(lsBoost, trainSet, regTreeFactory);
        optimizer.setShrinkage(config.getDouble("train.shrinkage"));
        optimizer.initialize();

        int progressInterval = config.getInt("train.showProgress.interval");

        int numIterations=config.getInt("train.numIterations");
        for (int i=1;i<=numIterations;i++){
            logger.info("iteration "+i);
            optimizer.iterate();
            if (config.getBoolean("train.showTrainProgress") && (i%progressInterval==0 || i==numIterations)){
                logger.info("training RMSE = "+ RMSE.rmse(lsBoost, trainSet));
            }
            if (config.getBoolean("train.showTestProgress") && (i%progressInterval==0 || i==numIterations)){
                logger.info("test RMSE = "+ RMSE.rmse(lsBoost, testSet));
            }
        }
        logger.info("training done!");
        String output = config.getString("output.folder");
        new File(output).mkdirs();
        File serializedModel =  new File(output,"model");
        Serialization.serialize(lsBoost, serializedModel);
        logger.info("model saved to "+serializedModel.getAbsolutePath());

        if(config.getBoolean("output.generatePMML")){
            File pmmlModel = new File(output, "model.pmml");
            PMMLConverter.savePMML(lsBoost,pmmlModel);
            logger.info("PMML model saved to "+pmmlModel.getAbsolutePath());

        }

        File reportFile = new File(output, "train_predictions.txt");
        report(lsBoost, trainSet, reportFile);
        logger.info("predictions on the training set are written to "+reportFile.getAbsolutePath());
    }

    private static void test(Config config, Logger logger) throws Exception{
        String output = config.getString("output.folder");
        File serializedModel =  new File(output,"model");
        LSBoost lsBoost = (LSBoost)Serialization.deserialize(serializedModel);
        String sparsity = config.getString("input.matrixType");
        DataSetType dataSetType = null;
        switch (sparsity){
            case "dense":
                dataSetType = DataSetType.REG_DENSE;
                break;
            case "sparse":
                dataSetType = DataSetType.REG_SPARSE;
                break;
            default:
                throw new IllegalArgumentException("input.matrixType should be dense or sparse");
        }
        RegDataSet testSet = TRECFormat.loadRegDataSet(config.getString("input.testData"),dataSetType, true);
        logger.info("test RMSE = "+ RMSE.rmse(lsBoost, testSet));
        File reportFile = new File(output, "test_predictions.txt");
        report(lsBoost, testSet, reportFile);
        logger.info("predictions on the test set are written to "+reportFile.getAbsolutePath());
    }

    private static void report(LSBoost lsBoost, RegDataSet dataSet, File reportFile) throws IOException {
        double[] prediction = lsBoost.predict(dataSet);
        String str = PrintUtil.toMutipleLines(prediction);
        FileUtils.writeStringToFile(reportFile, str);
    }



}
