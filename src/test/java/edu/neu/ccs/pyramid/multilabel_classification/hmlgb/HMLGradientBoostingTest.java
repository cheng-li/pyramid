package edu.neu.ccs.pyramid.multilabel_classification.hmlgb;


import edu.neu.ccs.pyramid.dataset.*;

import edu.neu.ccs.pyramid.eval.Accuracy;
import org.apache.commons.lang3.time.StopWatch;

import java.io.File;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;


public class HMLGradientBoostingTest {
    public static void main(String[] args) throws Exception{
//       spam_all();
//        test2_all();
        test3_all();
    }

    static void spam_all() throws Exception{
        spam_build();
        spam_load();
    }

    static void test2_all() throws Exception{
        test2_build();
        test2_load();
    }

    static void test3_all() throws Exception{
        test3_build();
        test3_load();
    }

    static void spam_load() throws Exception{
        ClfDataSet singleLabeldataSet = TRECFormat.loadClfDataSet("/Users/chengli/Datasets/spam/trec_data/test.trec",
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

        HMLGradientBoosting boosting = HMLGradientBoosting.deserialize("/Users/chengli/tmp/hmlgb/boosting.ser");
        System.out.println(Accuracy.accuracy(boosting,dataSet));
    }

    static void spam_build() throws Exception{


        ClfDataSet singleLabeldataSet = TRECFormat.loadClfDataSet("/Users/chengli/Datasets/spam/trec_data/train.trec",
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



        List<MultiLabel> assignments = new ArrayList<>();
        assignments.add(new MultiLabel(2).addLabel(0));
        assignments.add(new MultiLabel(2).addLabel(1));
        HMLGradientBoosting boosting = new HMLGradientBoosting(2,assignments);


        HMLGBConfig trainConfig = new HMLGBConfig.Builder(dataSet)
                .numLeaves(7).learningRate(0.1).numSplitIntervals(50).minDataPerLeaf(1)
                .dataSamplingRate(1).featureSamplingRate(1).build();
        System.out.println(Arrays.toString(trainConfig.getActiveFeatures()));

        boosting.setPriorProbs(dataSet,assignments);
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
//        for (int i=0;i<numDataPoints;i++){
//            FeatureRow featureRow = dataSet.getFeatureRow(i);
//            System.out.println("label="+dataSet.getMultiLabels()[i]);
//            System.out.println(boosting.calAssignmentScores(featureRow,assignments.get(0)));
//            System.out.println(boosting.calAssignmentScores(featureRow,assignments.get(1)));
//            System.out.println(boosting.predict(featureRow));
//        }
        System.out.println("accuracy");
        System.out.println(Accuracy.accuracy(boosting,dataSet));
        boosting.serialize("/Users/chengli/tmp/hmlgb/boosting.ser");

    }

    /**
     * add a fake label in spam data set, if x=spam and x_0<0.1, also label it as 2
     * @throws Exception
     */
    static void test2_build() throws Exception{


        ClfDataSet singleLabeldataSet = TRECFormat.loadClfDataSet("/Users/chengli/Datasets/spam/trec_data/train.trec",
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



        List<MultiLabel> assignments = new ArrayList<>();
        assignments.add(new MultiLabel(3).addLabel(0));
        assignments.add(new MultiLabel(3).addLabel(1));
        assignments.add(new MultiLabel(3).addLabel(1).addLabel(2));
        HMLGradientBoosting boosting = new HMLGradientBoosting(3,assignments);


        HMLGBConfig trainConfig = new HMLGBConfig.Builder(dataSet)
                .numLeaves(60).learningRate(0.1).numSplitIntervals(1000).minDataPerLeaf(2)
                .dataSamplingRate(1).featureSamplingRate(1).build();
        System.out.println(Arrays.toString(trainConfig.getActiveFeatures()));


        boosting.setPriorProbs(dataSet,assignments);
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
        boosting.serialize("/Users/chengli/tmp/hmlgb/boosting.ser");

    }

    static void test2_load() throws Exception{


        ClfDataSet singleLabeldataSet = TRECFormat.loadClfDataSet("/Users/chengli/Datasets/spam/trec_data/test.trec",
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


        HMLGradientBoosting boosting = HMLGradientBoosting.deserialize("/Users/chengli/tmp/hmlgb/boosting.ser");
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


        ClfDataSet singleLabeldataSet = TRECFormat.loadClfDataSet("/Users/chengli/Datasets/spam/trec_data/train.trec",
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



        List<MultiLabel> assignments = new ArrayList<>();
        assignments.add(new MultiLabel(4).addLabel(0));
        assignments.add(new MultiLabel(4).addLabel(1));
        assignments.add(new MultiLabel(4).addLabel(1).addLabel(2));
        assignments.add(new MultiLabel(4).addLabel(1).addLabel(3));
        assignments.add(new MultiLabel(4).addLabel(1).addLabel(2).addLabel(3));
        HMLGradientBoosting boosting = new HMLGradientBoosting(4,assignments);


        HMLGBConfig trainConfig = new HMLGBConfig.Builder(dataSet)
                .numLeaves(100).learningRate(0.1).numSplitIntervals(1000).minDataPerLeaf(2)
                .dataSamplingRate(1).featureSamplingRate(1).build();
        System.out.println(Arrays.toString(trainConfig.getActiveFeatures()));


        boosting.setPriorProbs(dataSet,assignments);
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
        boosting.serialize("/Users/chengli/tmp/hmlgb/boosting.ser");

    }

    static void test3_load() throws Exception{



        ClfDataSet singleLabeldataSet = TRECFormat.loadClfDataSet("/Users/chengli/Datasets/spam/trec_data/test.trec",
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


        HMLGradientBoosting boosting = HMLGradientBoosting.deserialize("/Users/chengli/tmp/hmlgb/boosting.ser");
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