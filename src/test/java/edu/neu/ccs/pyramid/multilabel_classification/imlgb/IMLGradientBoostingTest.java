package edu.neu.ccs.pyramid.multilabel_classification.imlgb;

import edu.neu.ccs.pyramid.configuration.Config;
import edu.neu.ccs.pyramid.dataset.*;
import edu.neu.ccs.pyramid.eval.Accuracy;
import edu.neu.ccs.pyramid.eval.Overlap;
import org.apache.commons.lang3.time.StopWatch;
import org.apache.mahout.math.Vector;

import java.io.File;
import java.util.*;
import java.util.Arrays;
import java.util.stream.Collectors;

public class IMLGradientBoostingTest {
    private static final Config config = new Config("configs/local.config");
    private static final String DATASETS = config.getString("input.datasets");
    private static final String TMP = config.getString("output.tmp");

    public static void main(String[] args) throws Exception{
//        test1();
        test4();
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
        MultiLabelClfDataSet dataSet = MLClfDataSetBuilder.getBuilder()
                .numDataPoints(numDataPoints).numFeatures(numFeatures)
                .numClasses(2).build();
        int[] labels = singleLabeldataSet.getLabels();
        for (int i=0;i<numDataPoints;i++){
            dataSet.addLabel(i,labels[i]);
            for (int j=0;j<numFeatures;j++){
                double value = singleLabeldataSet.getRow(i).get(j);
                dataSet.setFeatureValue(i,j,value);
            }
        }

        IMLGradientBoosting boosting = IMLGradientBoosting.deserialize(new File(TMP,"/imlgb/boosting.ser"));
        System.out.println(Accuracy.accuracy(boosting, dataSet));
//        System.out.println(IMLGBInspector.decisionProcess(boosting,dataSet.getRow(0),0));
        for (int i=0;i<10;i++){
            MultiLabel trueLabel = dataSet.getMultiLabels()[i];
            MultiLabel prediction = boosting.predict(dataSet.getRow(i));
            if (!prediction.equals(trueLabel)){
//                System.out.println(IMLGBInspector.analyzeMistake(boosting,dataSet.getRow(i),trueLabel,prediction,singleLabeldataSet.getSetting().getLabelTranslator(),10));
            }
        }
    }

    private static void spam_build() throws Exception{


        ClfDataSet singleLabeldataSet = TRECFormat.loadClfDataSet(new File(DATASETS,"/spam/trec_data/train.trec"),
                DataSetType.CLF_DENSE, true);
        int numDataPoints = singleLabeldataSet.getNumDataPoints();
        int numFeatures = singleLabeldataSet.getNumFeatures();
        MultiLabelClfDataSet dataSet = MLClfDataSetBuilder.getBuilder()
                .numDataPoints(numDataPoints).numFeatures(numFeatures)
                .numClasses(2).build();
        int[] labels = singleLabeldataSet.getLabels();
        for (int i=0;i<numDataPoints;i++){
            dataSet.addLabel(i,labels[i]);
            for (int j=0;j<numFeatures;j++){
                double value = singleLabeldataSet.getRow(i).get(j);
                dataSet.setFeatureValue(i,j,value);
            }
        }


        IMLGradientBoosting boosting = new IMLGradientBoosting(2);


        IMLGBConfig trainConfig = new IMLGBConfig.Builder(dataSet)
                .numLeaves(7).learningRate(0.1).numSplitIntervals(50).minDataPerLeaf(3)
                .dataSamplingRate(1).featureSamplingRate(1).build();
        System.out.println(Arrays.toString(trainConfig.getActiveFeatures()));

        IMLGBTrainer trainer = new IMLGBTrainer(trainConfig,boosting);

        StopWatch stopWatch = new StopWatch();
        stopWatch.start();
        for (int round =0;round<200;round++){
            System.out.println("round="+round);
            trainer.iterate();
            System.out.println("accuracy="+Accuracy.accuracy(boosting,dataSet));
//            System.out.println(Arrays.toString(boosting.getGradients(0)));
//            System.out.println(Arrays.toString(boosting.getGradients(1)));

        }
        stopWatch.stop();
        System.out.println(stopWatch);
        System.out.println(boosting);
        for (int i=0;i<numDataPoints;i++){
            org.apache.mahout.math.Vector featureRow = dataSet.getRow(i);
            System.out.println(""+i);
            System.out.println(dataSet.getMultiLabels()[i]);
            System.out.println(boosting.predict(featureRow));
        }
        System.out.println("accuracy");
        System.out.println(Accuracy.accuracy(boosting,dataSet));
        boosting.serialize(new File(TMP,"/imlgb/boosting.ser"));
        Comparator<Map.Entry<List<Integer>,Double>> comparator = Comparator.comparing(entry -> entry.getValue());
        System.out.println(IMLGBInspector.countPathMatches(boosting,dataSet,0).entrySet().stream().sorted(comparator.reversed()).collect(Collectors.toList()).get(0));
//        System.out.println(pathcount.values().stream().sorted().collect(Collectors.toList()));

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
        MultiLabelClfDataSet dataSet = MLClfDataSetBuilder.getBuilder()
                .numDataPoints(numDataPoints).numFeatures(numFeatures)
                .numClasses(3).build();
        int[] labels = singleLabeldataSet.getLabels();
        for (int i=0;i<numDataPoints;i++){
            dataSet.addLabel(i,labels[i]);
            if (labels[i]==1 && singleLabeldataSet.getRow(i).get(0)<0.1){
                dataSet.addLabel(i,2);
            }
            for (int j=0;j<numFeatures;j++){
                double value = singleLabeldataSet.getRow(i).get(j);
                dataSet.setFeatureValue(i,j,value);
            }
        }


        IMLGradientBoosting boosting = new IMLGradientBoosting(3);


        IMLGBConfig trainConfig = new IMLGBConfig.Builder(dataSet)
                .numLeaves(60).learningRate(0.1).numSplitIntervals(1000).minDataPerLeaf(2)
                .dataSamplingRate(1).featureSamplingRate(1).build();
        System.out.println(Arrays.toString(trainConfig.getActiveFeatures()));


        IMLGBTrainer trainer = new IMLGBTrainer(trainConfig,boosting);

        StopWatch stopWatch = new StopWatch();
        stopWatch.start();
        for (int round =0;round<20;round++){
            System.out.println("round="+round);
            trainer.iterate();
            System.out.println("accuracy="+Accuracy.accuracy(boosting,dataSet));
//            System.out.println(Arrays.toString(boosting.getGradients(0)));
//            System.out.println(Arrays.toString(boosting.getGradients(1)));

        }
        stopWatch.stop();
        System.out.println(stopWatch);
        System.out.println(boosting);
        for (int i=0;i<numDataPoints;i++){
            Vector featureRow = dataSet.getRow(i);
            MultiLabel label = dataSet.getMultiLabels()[i];
            MultiLabel prediction = boosting.predict(featureRow);
//            System.out.println("label="+label);
//            System.out.println(boosting.calAssignmentScore(featureRow,assignments.get(0)));
//            System.out.println(boosting.calAssignmentScore(featureRow,assignments.get(1)));
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
        MultiLabelClfDataSet dataSet = MLClfDataSetBuilder.getBuilder()
                .numDataPoints(numDataPoints).numFeatures(numFeatures)
                .numClasses(3).build();
        int[] labels = singleLabeldataSet.getLabels();
        for (int i=0;i<numDataPoints;i++){
            dataSet.addLabel(i,labels[i]);
            if (labels[i]==1 && singleLabeldataSet.getRow(i).get(0)<0.1){
                dataSet.addLabel(i,2);
            }
            for (int j=0;j<numFeatures;j++){
                double value = singleLabeldataSet.getRow(i).get(j);
                dataSet.setFeatureValue(i,j,value);
            }
        }


        IMLGradientBoosting boosting = IMLGradientBoosting.deserialize(new File(TMP,"/imlgb/boosting.ser"));
        System.out.println(Accuracy.accuracy(boosting,dataSet));
        for (int i=0;i<numDataPoints;i++){
            Vector featureRow = dataSet.getRow(i);
            MultiLabel label = dataSet.getMultiLabels()[i];
            MultiLabel prediction = boosting.predict(featureRow);
//            System.out.println("label="+label);
//            System.out.println(boosting.calAssignmentScore(featureRow,assignments.get(0)));
//            System.out.println(boosting.calAssignmentScore(featureRow,assignments.get(1)));
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
        MultiLabelClfDataSet dataSet = MLClfDataSetBuilder.getBuilder()
                .numDataPoints(numDataPoints).numFeatures(numFeatures)
                .numClasses(4).build();
        int[] labels = singleLabeldataSet.getLabels();
        for (int i=0;i<numDataPoints;i++){
            dataSet.addLabel(i,labels[i]);
            if (labels[i]==1 && singleLabeldataSet.getRow(i).get(0)<0.1){
                dataSet.addLabel(i,2);
            }
            if (labels[i]==1 && singleLabeldataSet.getRow(i).get(1)<0.1){
                dataSet.addLabel(i,3);
            }
            for (int j=0;j<numFeatures;j++){
                double value = singleLabeldataSet.getRow(i).get(j);
                dataSet.setFeatureValue(i,j,value);
            }
        }


        IMLGradientBoosting boosting = new IMLGradientBoosting(4);
        List<MultiLabel> assignments = DataSetUtil.gatherLabels(dataSet);
        boosting.setAssignments(assignments);

        IMLGBConfig trainConfig = new IMLGBConfig.Builder(dataSet)
                .numLeaves(10).learningRate(0.1).numSplitIntervals(1000).minDataPerLeaf(2)
                .dataSamplingRate(1).featureSamplingRate(1).build();
        System.out.println(Arrays.toString(trainConfig.getActiveFeatures()));

        IMLGBTrainer trainer = new IMLGBTrainer(trainConfig,boosting);

        StopWatch stopWatch = new StopWatch();
        stopWatch.start();
        for (int round =0;round<100;round++){
            System.out.println("round="+round);
            trainer.iterate();
            System.out.println("accuracy="+Accuracy.accuracy(boosting,dataSet));
//            System.out.println(Arrays.toString(boosting.getGradients(0)));
//            System.out.println(Arrays.toString(boosting.getGradients(1)));
//            System.out.println(Arrays.toString(boosting.getGradients(2)));
//            System.out.println(Arrays.toString(boosting.getGradients(3)));

        }
        stopWatch.stop();
        System.out.println(stopWatch);
//        System.out.println(boosting);
        for (int i=0;i<numDataPoints;i++){
            Vector featureRow = dataSet.getRow(i);
            MultiLabel label = dataSet.getMultiLabels()[i];
            MultiLabel prediction = boosting.predict(featureRow);
//            System.out.println("label="+label);
//            System.out.println(boosting.calAssignmentScore(featureRow,assignments.get(0)));
//            System.out.println(boosting.calAssignmentScore(featureRow,assignments.get(1)));
//            System.out.println("prediction="+prediction);
            if (!label.equals(prediction)){
                System.out.println(i);
                System.out.println("label="+label);
                System.out.println("prediction="+prediction);
            }
        }
        System.out.println("accuracy");
        System.out.println(Accuracy.accuracy(boosting,dataSet));
        System.out.println("overlap = "+ Overlap.overlap(boosting,dataSet));
        boosting.serialize(new File(TMP,"/imlgb/boosting.ser"));

    }

    static void test3_load() throws Exception{



        ClfDataSet singleLabeldataSet = TRECFormat.loadClfDataSet(new File(DATASETS,"/spam/trec_data/test.trec"),
                DataSetType.CLF_DENSE, true);
        int numDataPoints = singleLabeldataSet.getNumDataPoints();
        int numFeatures = singleLabeldataSet.getNumFeatures();
        MultiLabelClfDataSet dataSet = MLClfDataSetBuilder.getBuilder()
                .numDataPoints(numDataPoints).numFeatures(numFeatures)
                .numClasses(4).build();
        int[] labels = singleLabeldataSet.getLabels();
        for (int i=0;i<numDataPoints;i++){
            dataSet.addLabel(i,labels[i]);
            if (labels[i]==1 && singleLabeldataSet.getRow(i).get(0)<0.1){
                dataSet.addLabel(i,2);
            }
            if (labels[i]==1 && singleLabeldataSet.getRow(i).get(1)<0.1){
                dataSet.addLabel(i,3);
            }
            for (int j=0;j<numFeatures;j++){
                double value = singleLabeldataSet.getRow(i).get(j);
                dataSet.setFeatureValue(i,j,value);
            }
        }


        IMLGradientBoosting boosting = IMLGradientBoosting.deserialize(new File(TMP,"/imlgb/boosting.ser"));
        System.out.println("accuracy="+Accuracy.accuracy(boosting,dataSet));
        for (int i=0;i<numDataPoints;i++){
           Vector featureRow = dataSet.getRow(i);
            MultiLabel label = dataSet.getMultiLabels()[i];
            MultiLabel prediction = boosting.predict(featureRow);
//            System.out.println("label="+label);
//            System.out.println(boosting.calAssignmentScore(featureRow,assignments.get(0)));
//            System.out.println(boosting.calAssignmentScore(featureRow,assignments.get(1)));
//            System.out.println("prediction="+prediction);
//            if (!MultiLabel.equivalent(label,prediction)){
//                System.out.println(i);
//                System.out.println("label="+label);
//                System.out.println("prediction="+prediction);
//            }
        }

        System.out.println("overlap = "+ Overlap.overlap(boosting,dataSet));



    }

    static void test4() throws Exception{
        MultiLabelClfDataSet dataSet = TRECFormat.loadMultiLabelClfDataSet(new File(DATASETS, "ohsumed/3/train.trec"), DataSetType.ML_CLF_SPARSE, true);
        MultiLabelClfDataSet testSet = TRECFormat.loadMultiLabelClfDataSet(new File(DATASETS, "ohsumed/3/test.trec"), DataSetType.ML_CLF_SPARSE, true);
        IMLGradientBoosting boosting = new IMLGradientBoosting(dataSet.getNumClasses());
        List<MultiLabel> assignments = DataSetUtil.gatherLabels(dataSet);
        boosting.setAssignments(assignments);

        IMLGBConfig trainConfig = new IMLGBConfig.Builder(dataSet)
                .numLeaves(2).learningRate(0.1).numSplitIntervals(1000).minDataPerLeaf(2)
                .dataSamplingRate(1).featureSamplingRate(1).build();

        IMLGBTrainer trainer = new IMLGBTrainer(trainConfig,boosting);
        StopWatch stopWatch = new StopWatch();
        stopWatch.start();

        for (int round =0;round<100;round++){
            System.out.println("round="+round);
            trainer.iterate();
            System.out.println(stopWatch);
        }

        System.out.println("training accuracy="+ Accuracy.accuracy(boosting, dataSet));
        System.out.println("training overlap = "+ Overlap.overlap(boosting,dataSet));
        System.out.println("test accuracy="+ Accuracy.accuracy(boosting, testSet));
        System.out.println("test overlap = "+ Overlap.overlap(boosting,testSet));
    }

}