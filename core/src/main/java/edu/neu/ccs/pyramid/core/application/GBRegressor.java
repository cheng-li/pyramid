package edu.neu.ccs.pyramid.core.application;

import edu.neu.ccs.pyramid.core.configuration.Config;
import edu.neu.ccs.pyramid.core.dataset.RegDataSet;
import edu.neu.ccs.pyramid.core.dataset.TRECFormat;
import edu.neu.ccs.pyramid.core.eval.RMSE;
import edu.neu.ccs.pyramid.core.regression.least_squares_boost.LSBoost;
import edu.neu.ccs.pyramid.core.regression.least_squares_boost.LSBoostOptimizer;
import edu.neu.ccs.pyramid.core.regression.regression_tree.RegTreeFactory;
import edu.neu.ccs.pyramid.core.util.PrintUtil;
import edu.neu.ccs.pyramid.core.util.Serialization;
import edu.neu.ccs.pyramid.core.dataset.DataSetType;
import edu.neu.ccs.pyramid.core.regression.regression_tree.RegTreeConfig;
import org.apache.commons.io.FileUtils;

import java.io.File;
import java.io.IOException;

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
        System.out.println(config);

        if (config.getBoolean("train")){
            train(config);
        }

        if (config.getBoolean("test")){
            test(config);
        }
    }

    private static void train(Config config) throws Exception{
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
            System.out.println("iteration "+i);
            optimizer.iterate();
            if (config.getBoolean("train.showTrainProgress") && (i%progressInterval==0 || i==numIterations)){
                System.out.println("training RMSE = "+ RMSE.rmse(lsBoost, trainSet));
            }
            if (config.getBoolean("train.showTestProgress") && (i%progressInterval==0 || i==numIterations)){
                System.out.println("test RMSE = "+ RMSE.rmse(lsBoost, testSet));
            }
        }
        System.out.println("training done!");
        String output = config.getString("output.folder");
        new File(output).mkdirs();
        File serializedModel =  new File(output,"model");
        Serialization.serialize(lsBoost, serializedModel);
        System.out.println("model saved to "+serializedModel.getAbsolutePath());
        File reportFile = new File(output, "train_predictions.txt");
        report(lsBoost, trainSet, reportFile);
        System.out.println("predictions on the training set are written to "+reportFile.getAbsolutePath());
    }

    private static void test(Config config) throws Exception{
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
        System.out.println("test RMSE = "+ RMSE.rmse(lsBoost, testSet));
        File reportFile = new File(output, "test_predictions.txt");
        report(lsBoost, testSet, reportFile);
        System.out.println("predictions on the test set are written to "+reportFile.getAbsolutePath());
    }

    private static void report(LSBoost lsBoost, RegDataSet dataSet, File reportFile) throws IOException {
        double[] prediction = lsBoost.predict(dataSet);
        String str = PrintUtil.toMutipleLines(prediction);
        FileUtils.writeStringToFile(reportFile, str);
    }



}
