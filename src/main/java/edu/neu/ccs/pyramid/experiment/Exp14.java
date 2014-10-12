package edu.neu.ccs.pyramid.experiment;

import edu.neu.ccs.pyramid.configuration.Config;
import edu.neu.ccs.pyramid.dataset.*;
import edu.neu.ccs.pyramid.eval.Accuracy;
import edu.neu.ccs.pyramid.multilabel_classification.imlgb.IMLGBConfig;
import edu.neu.ccs.pyramid.multilabel_classification.imlgb.IMLGradientBoosting;
import org.apache.commons.lang3.time.StopWatch;

import java.io.File;
import java.util.List;

/**
 * imlgb
 * Created by chengli on 10/11/14.
 */
public class Exp14 {
    public static void main(String[] args) throws Exception{
        if (args.length !=1){
            throw new IllegalArgumentException("please specify the config file");
        }

        Config config = new Config(args[0]);
        System.out.println(config);

        if (config.getBoolean("train")){
            train(config);
        }
        if (config.getBoolean("verify")){
            verify(config);
        }
        if (config.getBoolean("test")){
            test(config);
        }



    }

    static MultiLabelClfDataSet loadTrainData(Config config) throws Exception{
        String trainFile = new File(config.getString("input.folder"),
                config.getString("input.trainData")).getAbsolutePath();
        MultiLabelClfDataSet dataSet;

        if (config.getBoolean("featureMatrix.sparse")){
            dataSet= TRECFormat.loadMultiLabelClfDataSet(new File(trainFile), DataSetType.ML_CLF_SPARSE,
                    true);
        } else {
            dataSet= TRECFormat.loadMultiLabelClfDataSet(new File(trainFile), DataSetType.ML_CLF_DENSE,
                    true);
        }

        return dataSet;
    }

    static MultiLabelClfDataSet loadTestData(Config config) throws Exception{
        String trainFile = new File(config.getString("input.folder"),
                config.getString("input.testData")).getAbsolutePath();
        MultiLabelClfDataSet dataSet;

        if (config.getBoolean("featureMatrix.sparse")){
            dataSet= TRECFormat.loadMultiLabelClfDataSet(new File(trainFile), DataSetType.ML_CLF_SPARSE,
                    true);
        } else {
            dataSet= TRECFormat.loadMultiLabelClfDataSet(new File(trainFile), DataSetType.ML_CLF_DENSE,
                    true);
        }

        return dataSet;
    }

    static void train(Config config) throws Exception{
        String archive = config.getString("archive.folder");
        int numIterations = config.getInt("train.numIterations");
        int numLeaves = config.getInt("train.numLeaves");
        double learningRate = config.getDouble("train.learningRate");
        int minDataPerLeaf = config.getInt("train.minDataPerLeaf");
        String modelName = config.getString("archive.model");
        double featureSamplingRate = config.getDouble("train.featureSamplingRate");
        double dataSamplingRate = config.getDouble("train.dataSamplingRate");

        StopWatch stopWatch = new StopWatch();
        stopWatch.start();

        MultiLabelClfDataSet dataSet = loadTrainData(config);
        int numClasses = dataSet.getNumClasses();
        System.out.println("number of class = "+numClasses);
        IMLGBConfig imlgbConfig = new IMLGBConfig.Builder(dataSet)
                .dataSamplingRate(dataSamplingRate)
                .featureSamplingRate(featureSamplingRate)
                .learningRate(learningRate)
                .minDataPerLeaf(minDataPerLeaf)
                .numLeaves(numLeaves)
                .numSplitIntervals(config.getInt("train.numSplitIntervals"))
                .build();

        IMLGradientBoosting boosting = new IMLGradientBoosting(numClasses);
        List<MultiLabel> assignments = DataSetUtil.gatherLabels(dataSet);
        boosting.setAssignments(assignments);
        boosting.setPriorProbs(dataSet);
        boosting.setTrainConfig(imlgbConfig);

        for (int i=0;i<numIterations;i++){
            System.out.println("iteration "+i);
            boosting.boostOneRound();
            System.out.println("accuracy on training set = "+ Accuracy.accuracy(boosting,
                    dataSet));
        }
        File serializedModel =  new File(archive,modelName);


        boosting.serialize(serializedModel);
        System.out.println(stopWatch);

    }

    static void verify(Config config) throws Exception{
        String archive = config.getString("archive.folder");
        String modelName = config.getString("archive.model");

        IMLGradientBoosting boosting = IMLGradientBoosting.deserialize(new File(archive,modelName));
        MultiLabelClfDataSet dataSet = loadTrainData(config);
        System.out.println("accuracy on training set = "+Accuracy.accuracy(boosting,dataSet));
        for (int i=0;i<dataSet.getNumDataPoints();i++){
            System.out.println(""+i);
            System.out.println("true labels:");
            System.out.println(dataSet.getMultiLabels()[i]);
            System.out.println("predictions:");
            System.out.println(boosting.predict(dataSet.getFeatureRow(i)));
        }
    }

    static void test(Config config) throws Exception{
        String archive = config.getString("archive.folder");
        String modelName = config.getString("archive.model");

        IMLGradientBoosting boosting = IMLGradientBoosting.deserialize(new File(archive,modelName));
        MultiLabelClfDataSet dataSet = loadTestData(config);
        System.out.println("accuracy on test set = "+Accuracy.accuracy(boosting,dataSet));
        for (int i=0;i<dataSet.getNumDataPoints();i++){
            System.out.println(""+i);
            System.out.println("true labels:");
            System.out.println(dataSet.getMultiLabels()[i]);
            System.out.println("predictions:");
            System.out.println(boosting.predict(dataSet.getFeatureRow(i)));
        }
    }
}
