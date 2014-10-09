package edu.neu.ccs.pyramid.multilabel_classification.imlgb;

import edu.neu.ccs.pyramid.configuration.Config;
import edu.neu.ccs.pyramid.dataset.*;
import edu.neu.ccs.pyramid.eval.Accuracy;
import org.apache.commons.lang3.time.StopWatch;

import java.io.File;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import static org.junit.Assert.*;

public class IMLGradientBoostingTest {
    private static final Config config = new Config("configs/local.config");
    private static final String DATASETS = config.getString("input.datasets");
    private static final String TMP = config.getString("output.tmp");

    public static void main(String[] args) throws Exception{
        test3();
    }

    private static void test1() throws Exception{
        spam_build();
        spam_load();
    }

    private static void test2() throws Exception{
        test2_build();
        test2_load();
    }

    private static void test3() throws Exception{
        test3_build();
        test3_load();
    }

    private static void spam_load() throws Exception{
        ClfDataSet singleLabeldataSet = TRECFormat.loadClfDataSet(new File(DATASETS,"/spam/trec_data/test.trec"),
                DataSetType.CLF_DENSE, true);
        int numDataPoints = singleLabeldataSet.getNumDataPoints();
        int numFeatures = singleLabeldataSet.getNumFeatures();
        MultiLabelClfDataSet dataSet = new DenseMLClfDataSet(numDataPoints,
                numFeatures,2);
        int[] labels = singleLabeldataSet.getLabels();
        for (int i=0;i<numDataPoints;i++){
            dataSet.addLabel(i,labels[i]);
            for (int j=0;j<numFeatures;j++){
                double value = singleLabeldataSet.getFeatureRow(i).getVector().get(j);
                dataSet.setFeatureValue(i,j,value);
            }
        }

        IMLGradientBoosting boosting = IMLGradientBoosting.deserialize(new File(TMP,"/imlgb/boosting.ser"));
        System.out.println(Accuracy.accuracy(boosting, dataSet));
    }

    private static void spam_build() throws Exception{


        ClfDataSet singleLabeldataSet = TRECFormat.loadClfDataSet(new File(DATASETS,"/spam/trec_data/train.trec"),
                DataSetType.CLF_DENSE, true);
        int numDataPoints = singleLabeldataSet.getNumDataPoints();
        int numFeatures = singleLabeldataSet.getNumFeatures();
        MultiLabelClfDataSet dataSet = new DenseMLClfDataSet(numDataPoints,
                numFeatures,2);
        int[] labels = singleLabeldataSet.getLabels();
        for (int i=0;i<numDataPoints;i++){
            dataSet.addLabel(i,labels[i]);
            for (int j=0;j<numFeatures;j++){
                double value = singleLabeldataSet.getFeatureRow(i).getVector().get(j);
                dataSet.setFeatureValue(i,j,value);
            }
        }


        IMLGradientBoosting boosting = new IMLGradientBoosting(2);


        IMLGBConfig trainConfig = new IMLGBConfig.Builder(dataSet)
                .numLeaves(7).learningRate(0.1).numSplitIntervals(50).minDataPerLeaf(1)
                .dataSamplingRate(1).featureSamplingRate(1).build();
        System.out.println(Arrays.toString(trainConfig.getActiveFeatures()));

        boosting.setPriorProbs(dataSet);
        boosting.setTrainConfig(trainConfig);

        StopWatch stopWatch = new StopWatch();
        stopWatch.start();
        for (int round =0;round<200;round++){
            System.out.println("round="+round);
            boosting.boostOneRound();
            System.out.println("accuracy="+Accuracy.accuracy(boosting,dataSet));
//            System.out.println(Arrays.toString(boosting.getGradients(0)));
//            System.out.println(Arrays.toString(boosting.getGradients(1)));

        }
        stopWatch.stop();
        System.out.println(stopWatch);
        System.out.println(boosting);
        for (int i=0;i<numDataPoints;i++){
            FeatureRow featureRow = dataSet.getFeatureRow(i);
            System.out.println(""+i);
            System.out.println(dataSet.getMultiLabels()[i]);
            System.out.println(boosting.predict(featureRow));
        }
        System.out.println("accuracy");
        System.out.println(Accuracy.accuracy(boosting,dataSet));
        boosting.serialize(new File(TMP,"/imlgb/boosting.ser"));

    }

    /**
     * add a fake label in spam data set, if x=spam and x_0<0.1, also label it as 2
     * @throws Exception
     */
    static void test2_build() throws Exception{


        ClfDataSet singleLabeldataSet = TRECFormat.loadClfDataSet(new File(DATASETS,"/spam/trec_data/train.trec"),
                DataSetType.CLF_DENSE, true);
        int numDataPoints = singleLabeldataSet.getNumDataPoints();
        int numFeatures = singleLabeldataSet.getNumFeatures();
        MultiLabelClfDataSet dataSet = new DenseMLClfDataSet(numDataPoints,
                numFeatures,3);
        int[] labels = singleLabeldataSet.getLabels();
        for (int i=0;i<numDataPoints;i++){
            dataSet.addLabel(i,labels[i]);
            if (labels[i]==1 && singleLabeldataSet.getFeatureRow(i).getVector().get(0)<0.1){
                dataSet.addLabel(i,2);
            }
            for (int j=0;j<numFeatures;j++){
                double value = singleLabeldataSet.getFeatureRow(i).getVector().get(j);
                dataSet.setFeatureValue(i,j,value);
            }
        }


        IMLGradientBoosting boosting = new IMLGradientBoosting(3);


        IMLGBConfig trainConfig = new IMLGBConfig.Builder(dataSet)
                .numLeaves(60).learningRate(0.1).numSplitIntervals(1000).minDataPerLeaf(2)
                .dataSamplingRate(1).featureSamplingRate(1).build();
        System.out.println(Arrays.toString(trainConfig.getActiveFeatures()));


        boosting.setPriorProbs(dataSet);
        boosting.setTrainConfig(trainConfig);

        StopWatch stopWatch = new StopWatch();
        stopWatch.start();
        for (int round =0;round<20;round++){
            System.out.println("round="+round);
            boosting.boostOneRound();
            System.out.println("accuracy="+Accuracy.accuracy(boosting,dataSet));
//            System.out.println(Arrays.toString(boosting.getGradients(0)));
//            System.out.println(Arrays.toString(boosting.getGradients(1)));

        }
        stopWatch.stop();
        System.out.println(stopWatch);
        System.out.println(boosting);
        for (int i=0;i<numDataPoints;i++){
            FeatureRow featureRow = dataSet.getFeatureRow(i);
            MultiLabel label = dataSet.getMultiLabels()[i];
            MultiLabel prediction = boosting.predict(featureRow);
//            System.out.println("label="+label);
//            System.out.println(boosting.calAssignmentScores(featureRow,assignments.get(0)));
//            System.out.println(boosting.calAssignmentScores(featureRow,assignments.get(1)));
//            System.out.println("prediction="+prediction);
//            if (!MultiLabel.equivalent(label,prediction)){
//                System.out.println(i);
//                System.out.println("label="+label);
//                System.out.println("prediction="+prediction);
//            }
        }
        System.out.println("accuracy");
        System.out.println(Accuracy.accuracy(boosting,dataSet));
        boosting.serialize(new File(TMP,"imlgb/boosting.ser"));

    }

    static void test2_load() throws Exception{


        ClfDataSet singleLabeldataSet = TRECFormat.loadClfDataSet(new File(DATASETS,"spam/trec_data/test.trec"),
                DataSetType.CLF_DENSE, true);
        int numDataPoints = singleLabeldataSet.getNumDataPoints();
        int numFeatures = singleLabeldataSet.getNumFeatures();
        MultiLabelClfDataSet dataSet = new DenseMLClfDataSet(numDataPoints,
                numFeatures,3);
        int[] labels = singleLabeldataSet.getLabels();
        for (int i=0;i<numDataPoints;i++){
            dataSet.addLabel(i,labels[i]);
            if (labels[i]==1 && singleLabeldataSet.getFeatureRow(i).getVector().get(0)<0.1){
                dataSet.addLabel(i,2);
            }
            for (int j=0;j<numFeatures;j++){
                double value = singleLabeldataSet.getFeatureRow(i).getVector().get(j);
                dataSet.setFeatureValue(i,j,value);
            }
        }


        IMLGradientBoosting boosting = IMLGradientBoosting.deserialize(new File(TMP,"/imlgb/boosting.ser"));
        System.out.println(Accuracy.accuracy(boosting,dataSet));
        for (int i=0;i<numDataPoints;i++){
            FeatureRow featureRow = dataSet.getFeatureRow(i);
            MultiLabel label = dataSet.getMultiLabels()[i];
            MultiLabel prediction = boosting.predict(featureRow);
//            System.out.println("label="+label);
//            System.out.println(boosting.calAssignmentScores(featureRow,assignments.get(0)));
//            System.out.println(boosting.calAssignmentScores(featureRow,assignments.get(1)));
//            System.out.println("prediction="+prediction);
//            if (!MultiLabel.equivalent(label,prediction)){
//                System.out.println(i);
//                System.out.println("label="+label);
//                System.out.println("prediction="+prediction);
//            }
        }
    }


    /**
     * add 2 fake labels in spam data set,
     * if x=spam and x_0<0.1, also label it as 2
     * if x=spam and x_1<0.1, also label it as 3
     * @throws Exception
     */
    static void test3_build() throws Exception{


        ClfDataSet singleLabeldataSet = TRECFormat.loadClfDataSet(new File(DATASETS,"spam/trec_data/train.trec"),
                DataSetType.CLF_DENSE, true);
        int numDataPoints = singleLabeldataSet.getNumDataPoints();
        int numFeatures = singleLabeldataSet.getNumFeatures();
        MultiLabelClfDataSet dataSet = new DenseMLClfDataSet(numDataPoints,
                numFeatures,4);
        int[] labels = singleLabeldataSet.getLabels();
        for (int i=0;i<numDataPoints;i++){
            dataSet.addLabel(i,labels[i]);
            if (labels[i]==1 && singleLabeldataSet.getFeatureRow(i).getVector().get(0)<0.1){
                dataSet.addLabel(i,2);
            }
            if (labels[i]==1 && singleLabeldataSet.getFeatureRow(i).getVector().get(1)<0.1){
                dataSet.addLabel(i,3);
            }
            for (int j=0;j<numFeatures;j++){
                double value = singleLabeldataSet.getFeatureRow(i).getVector().get(j);
                dataSet.setFeatureValue(i,j,value);
            }
        }


        IMLGradientBoosting boosting = new IMLGradientBoosting(4);


        IMLGBConfig trainConfig = new IMLGBConfig.Builder(dataSet)
                .numLeaves(100).learningRate(0.1).numSplitIntervals(1000).minDataPerLeaf(2)
                .dataSamplingRate(1).featureSamplingRate(1).build();
        System.out.println(Arrays.toString(trainConfig.getActiveFeatures()));


        boosting.setPriorProbs(dataSet);
        boosting.setTrainConfig(trainConfig);

        StopWatch stopWatch = new StopWatch();
        stopWatch.start();
        for (int round =0;round<10;round++){
            System.out.println("round="+round);
            boosting.boostOneRound();
            System.out.println("accuracy="+Accuracy.accuracy(boosting,dataSet));
            System.out.println(Arrays.toString(boosting.getGradients(0)));
            System.out.println(Arrays.toString(boosting.getGradients(1)));
            System.out.println(Arrays.toString(boosting.getGradients(2)));
            System.out.println(Arrays.toString(boosting.getGradients(3)));

        }
        stopWatch.stop();
        System.out.println(stopWatch);
//        System.out.println(boosting);
        for (int i=0;i<numDataPoints;i++){
            FeatureRow featureRow = dataSet.getFeatureRow(i);
            MultiLabel label = dataSet.getMultiLabels()[i];
            MultiLabel prediction = boosting.predict(featureRow);
//            System.out.println("label="+label);
//            System.out.println(boosting.calAssignmentScores(featureRow,assignments.get(0)));
//            System.out.println(boosting.calAssignmentScores(featureRow,assignments.get(1)));
//            System.out.println("prediction="+prediction);
            if (!MultiLabel.equivalent(label,prediction)){
                System.out.println(i);
                System.out.println("label="+label);
                System.out.println("prediction="+prediction);
            }
        }
        System.out.println("accuracy");
        System.out.println(Accuracy.accuracy(boosting,dataSet));
        boosting.serialize(new File(TMP,"/imlgb/boosting.ser"));

    }

    static void test3_load() throws Exception{



        ClfDataSet singleLabeldataSet = TRECFormat.loadClfDataSet(new File(DATASETS,"/spam/trec_data/test.trec"),
                DataSetType.CLF_DENSE, true);
        int numDataPoints = singleLabeldataSet.getNumDataPoints();
        int numFeatures = singleLabeldataSet.getNumFeatures();
        MultiLabelClfDataSet dataSet = new DenseMLClfDataSet(numDataPoints,
                numFeatures,4);
        int[] labels = singleLabeldataSet.getLabels();
        for (int i=0;i<numDataPoints;i++){
            dataSet.addLabel(i,labels[i]);
            if (labels[i]==1 && singleLabeldataSet.getFeatureRow(i).getVector().get(0)<0.1){
                dataSet.addLabel(i,2);
            }
            if (labels[i]==1 && singleLabeldataSet.getFeatureRow(i).getVector().get(1)<0.1){
                dataSet.addLabel(i,3);
            }
            for (int j=0;j<numFeatures;j++){
                double value = singleLabeldataSet.getFeatureRow(i).getVector().get(j);
                dataSet.setFeatureValue(i,j,value);
            }
        }


        IMLGradientBoosting boosting = IMLGradientBoosting.deserialize(new File(TMP,"/imlgb/boosting.ser"));
        System.out.println(Accuracy.accuracy(boosting,dataSet));
        for (int i=0;i<numDataPoints;i++){
            FeatureRow featureRow = dataSet.getFeatureRow(i);
            MultiLabel label = dataSet.getMultiLabels()[i];
            MultiLabel prediction = boosting.predict(featureRow);
//            System.out.println("label="+label);
//            System.out.println(boosting.calAssignmentScores(featureRow,assignments.get(0)));
//            System.out.println(boosting.calAssignmentScores(featureRow,assignments.get(1)));
//            System.out.println("prediction="+prediction);
//            if (!MultiLabel.equivalent(label,prediction)){
//                System.out.println(i);
//                System.out.println("label="+label);
//                System.out.println("prediction="+prediction);
//            }
        }




    }

}