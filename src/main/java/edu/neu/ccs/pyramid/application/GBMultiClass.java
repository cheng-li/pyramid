package edu.neu.ccs.pyramid.application;

import edu.neu.ccs.pyramid.classification.lkboost.LKBOutputCalculator;
import edu.neu.ccs.pyramid.classification.lkboost.LKBoost;
import edu.neu.ccs.pyramid.classification.lkboost.LKBoostOptimizer;
import edu.neu.ccs.pyramid.configuration.Config;
import edu.neu.ccs.pyramid.dataset.ClfDataSet;
import edu.neu.ccs.pyramid.dataset.DataSetType;
import edu.neu.ccs.pyramid.dataset.TRECFormat;
import edu.neu.ccs.pyramid.eval.Accuracy;
import edu.neu.ccs.pyramid.regression.regression_tree.RegTreeConfig;
import edu.neu.ccs.pyramid.regression.regression_tree.RegTreeFactory;
import edu.neu.ccs.pyramid.util.PrintUtil;
import edu.neu.ccs.pyramid.util.Serialization;
import org.apache.commons.io.FileUtils;

import java.io.File;
import java.io.IOException;

/**
 * multi-class gradient boosting with KL divergence loss
 * Created by chengli on 9/22/16.
 */
public class GBMultiClass {
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
                dataSetType = DataSetType.CLF_DENSE;
                break;
            case "sparse":
                dataSetType = DataSetType.CLF_SPARSE;
                break;
            default:
                throw new IllegalArgumentException("input.matrixType should be dense or sparse");
        }
        ClfDataSet trainSet = TRECFormat.loadClfDataSet(config.getString("input.trainData"),dataSetType, true);

        ClfDataSet testSet = null;
        if (config.getBoolean("train.showTestProgress")){
            testSet = TRECFormat.loadClfDataSet(config.getString("input.testData"),dataSetType, true);
        }

        int numClasses = trainSet.getNumClasses();
        LKBoost lkBoost = new LKBoost(numClasses);

        RegTreeConfig regTreeConfig = new RegTreeConfig().setMaxNumLeaves(config.getInt("train.numLeaves"));
        RegTreeFactory regTreeFactory = new RegTreeFactory(regTreeConfig);
        regTreeFactory.setLeafOutputCalculator(new LKBOutputCalculator(numClasses));
        LKBoostOptimizer optimizer = new LKBoostOptimizer(lkBoost, trainSet, regTreeFactory);
        optimizer.setShrinkage(config.getDouble("train.shrinkage"));
        optimizer.initialize();

        int progressInterval = config.getInt("train.showProgress.interval");

        int numIterations=config.getInt("train.numIterations");
        for (int i=1;i<=numIterations;i++){
            System.out.println("iteration "+i);
            optimizer.iterate();
            if (config.getBoolean("train.showTrainProgress") && (i%progressInterval==0 || i==numIterations)){
                System.out.println("training accuracy = "+ Accuracy.accuracy(lkBoost, trainSet));
            }
            if (config.getBoolean("train.showTestProgress") && (i%progressInterval==0 || i==numIterations)){
                System.out.println("test accuracy = "+ Accuracy.accuracy(lkBoost, testSet));
            }
        }
        System.out.println("training done!");
        String output = config.getString("output.folder");
        new File(output).mkdirs();
        File serializedModel =  new File(output,"model");
        Serialization.serialize(lkBoost, serializedModel);
        System.out.println("model saved to "+serializedModel.getAbsolutePath());
        File reportFile = new File(output, "train_predictions.txt");
        report(lkBoost, trainSet, reportFile);
        System.out.println("predictions on the training set are written to "+reportFile.getAbsolutePath());
    }

    private static void test(Config config) throws Exception{
        String output = config.getString("output.folder");
        File serializedModel =  new File(output,"model");
        LKBoost lkBoost = (LKBoost)Serialization.deserialize(serializedModel);
        String sparsity = config.getString("input.matrixType");
        DataSetType dataSetType = null;
        switch (sparsity){
            case "dense":
                dataSetType = DataSetType.CLF_DENSE;
                break;
            case "sparse":
                dataSetType = DataSetType.CLF_SPARSE;
                break;
            default:
                throw new IllegalArgumentException("input.matrixType should be dense or sparse");
        }
        ClfDataSet testSet = TRECFormat.loadClfDataSet(config.getString("input.testData"),dataSetType, true);
        System.out.println("test accuracy = "+ Accuracy.accuracy(lkBoost, testSet));
        File reportFile = new File(output, "test_predictions.txt");
        report(lkBoost, testSet, reportFile);
        System.out.println("predictions on the test set are written to "+reportFile.getAbsolutePath());
    }

    private static void report(LKBoost lkBoost, ClfDataSet dataSet, File reportFile) throws IOException {
        int[] prediction = lkBoost.predict(dataSet);
        String str = PrintUtil.toMutipleLines(prediction);
        FileUtils.writeStringToFile(reportFile, str);
    }
}
