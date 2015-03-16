package edu.neu.ccs.pyramid.multilabel_classification.hmlgb;


import com.fasterxml.jackson.databind.ObjectMapper;
import edu.neu.ccs.pyramid.configuration.Config;
import edu.neu.ccs.pyramid.dataset.*;

import edu.neu.ccs.pyramid.eval.Accuracy;
import edu.neu.ccs.pyramid.eval.MacroAveragedMeasures;
import edu.neu.ccs.pyramid.eval.Overlap;
import edu.neu.ccs.pyramid.multilabel_classification.MultiLabelPredictionAnalysis;
import edu.neu.ccs.pyramid.multilabel_classification.imlgb.IMLGBConfig;
import edu.neu.ccs.pyramid.multilabel_classification.imlgb.IMLGBTrainer;
import edu.neu.ccs.pyramid.multilabel_classification.imlgb.IMLGradientBoosting;
import org.apache.commons.lang3.time.StopWatch;
import org.apache.mahout.math.Vector;

import java.io.File;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;


public class HMLGradientBoostingTest {
    private static final Config config = new Config("configs/local.config");
    private static final String DATASETS = config.getString("input.datasets");
    private static final String TMP = config.getString("output.tmp");
    public static void main(String[] args) throws Exception{
//       spam_all();
//        test2_all();
//        test3_all();
//        test4();
//        test3_load();
        test5();
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

        HMLGradientBoosting boosting = HMLGradientBoosting.deserialize(new File(TMP,"/hmlgb/boosting.ser"));
        System.out.println("accuracy="+Accuracy.accuracy(boosting,dataSet));
        System.out.println("macro-averaged:");
        System.out.println(new MacroAveragedMeasures(boosting,dataSet));
    }

    static void spam_build() throws Exception{


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



        List<MultiLabel> assignments = new ArrayList<>();
        assignments.add(new MultiLabel().addLabel(0));
        assignments.add(new MultiLabel().addLabel(1));
        HMLGradientBoosting boosting = new HMLGradientBoosting(2,assignments);


        HMLGBConfig trainConfig = new HMLGBConfig.Builder(dataSet)
                .numLeaves(7).learningRate(0.1).numSplitIntervals(50).minDataPerLeaf(1)
                .dataSamplingRate(1).featureSamplingRate(1).build();
        System.out.println(Arrays.toString(trainConfig.getActiveFeatures()));

        HMLGBTrainer trainer = new HMLGBTrainer(trainConfig,boosting);

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
//        for (int i=0;i<numDataPoints;i++){
//            FeatureRow featureRow = dataSet.getRow(i);
//            System.out.println("label="+dataSet.getMultiLabels()[i]);
//            System.out.println(boosting.calAssignmentScores(featureRow,assignments.get(0)));
//            System.out.println(boosting.calAssignmentScores(featureRow,assignments.get(1)));
//            System.out.println(boosting.predict(featureRow));
//        }
        System.out.println("accuracy");
        System.out.println(Accuracy.accuracy(boosting,dataSet));
        System.out.println("macro-averaged:");
        System.out.println(new MacroAveragedMeasures(boosting,dataSet));



        boosting.serialize(new File(TMP,"/hmlgb/boosting.ser"));

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



        List<MultiLabel> assignments = new ArrayList<>();
        assignments.add(new MultiLabel().addLabel(0));
        assignments.add(new MultiLabel().addLabel(1));
        assignments.add(new MultiLabel().addLabel(1).addLabel(2));
        HMLGradientBoosting boosting = new HMLGradientBoosting(3,assignments);


        HMLGBConfig trainConfig = new HMLGBConfig.Builder(dataSet)
                .numLeaves(10).learningRate(0.1).numSplitIntervals(1000).minDataPerLeaf(2)
                .dataSamplingRate(1).featureSamplingRate(1).build();
        System.out.println(Arrays.toString(trainConfig.getActiveFeatures()));


        HMLGBTrainer trainer = new HMLGBTrainer(trainConfig,boosting);

        StopWatch stopWatch = new StopWatch();
        stopWatch.start();
        for (int round =0;round<30;round++){
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
        boosting.serialize(new File(TMP,"/hmlgb/boosting.ser"));

    }

    static void test2_load() throws Exception{


        ClfDataSet singleLabeldataSet = TRECFormat.loadClfDataSet(new File(DATASETS,"/spam/trec_data/test.trec"),
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


        HMLGradientBoosting boosting = HMLGradientBoosting.deserialize(new File(TMP,"/hmlgb/boosting.ser"));
        System.out.println(Accuracy.accuracy(boosting,dataSet));
        for (int i=0;i<numDataPoints;i++){
            Vector featureRow = dataSet.getRow(i);
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


        List<String> extLabels = new ArrayList<>();
        extLabels.add("non_spam");
        extLabels.add("spam");
        extLabels.add("fake2");
        extLabels.add("fake3");
        LabelTranslator labelTranslator = new LabelTranslator(extLabels);
        dataSet.setLabelTranslator(labelTranslator);

        List<MultiLabel> assignments = new ArrayList<>();
        assignments.add(new MultiLabel().addLabel(0));
        assignments.add(new MultiLabel().addLabel(1));
        assignments.add(new MultiLabel().addLabel(1).addLabel(2));
        assignments.add(new MultiLabel().addLabel(1).addLabel(3));
        assignments.add(new MultiLabel().addLabel(1).addLabel(2).addLabel(3));
        HMLGradientBoosting boosting = new HMLGradientBoosting(4,assignments);


        HMLGBConfig trainConfig = new HMLGBConfig.Builder(dataSet)
                .numLeaves(2).learningRate(0.1).numSplitIntervals(1000).minDataPerLeaf(2)
                .dataSamplingRate(1).featureSamplingRate(1).build();
        System.out.println(Arrays.toString(trainConfig.getActiveFeatures()));


        HMLGBTrainer trainer = new HMLGBTrainer(trainConfig,boosting);

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
//            System.out.println(boosting.calAssignmentScores(featureRow,assignments.get(0)));
//            System.out.println(boosting.calAssignmentScores(featureRow,assignments.get(1)));
//            System.out.println("prediction="+prediction);
            if (!label.equals(prediction)){
                System.out.println(i);
                System.out.println("label="+label);
                System.out.println("prediction="+prediction);
            }
        }
        System.out.println("accuracy");
        System.out.println(Accuracy.accuracy(boosting,dataSet));
        boosting.serialize(new File(TMP,"/hmlgb/boosting.ser"));


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

        List<String> extLabels = new ArrayList<>();
        extLabels.add("non_spam");
        extLabels.add("spam");
        extLabels.add("fake2");
        extLabels.add("fake3");
        LabelTranslator labelTranslator = new LabelTranslator(extLabels);
        dataSet.setLabelTranslator(labelTranslator);

        HMLGradientBoosting boosting = HMLGradientBoosting.deserialize(new File(TMP,"/hmlgb/boosting.ser"));
        System.out.println(Accuracy.accuracy(boosting,dataSet));
        for (int i=0;i<numDataPoints;i++){
            Vector featureRow = dataSet.getRow(i);
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

        MultiLabelPredictionAnalysis analysis = HMLGBInspector.analyzePrediction(boosting, dataSet, 0, 10);
        ObjectMapper mapper1 = new ObjectMapper();
        mapper1.writeValue(new File(TMP,"prediction_analysis.json"), analysis);

    }

    private static void test4() throws Exception{
        test4_build();
        test4_load();
    }

    /**
     * same as test3, the only difference is we now load data directly
     * @throws Exception
     */
    static void test4_build() throws Exception{
        MultiLabelClfDataSet dataSet = TRECFormat.loadMultiLabelClfDataSet(new File(DATASETS,"spam/4labels/train.trec"),
                DataSetType.ML_CLF_DENSE,true);

        List<MultiLabel> assignments = new ArrayList<>();
        assignments.add(new MultiLabel().addLabel(0));
        assignments.add(new MultiLabel().addLabel(1));
        assignments.add(new MultiLabel().addLabel(1).addLabel(2));
        assignments.add(new MultiLabel().addLabel(1).addLabel(3));
        assignments.add(new MultiLabel().addLabel(1).addLabel(2).addLabel(3));
        HMLGradientBoosting boosting = new HMLGradientBoosting(4,assignments);


        HMLGBConfig trainConfig = new HMLGBConfig.Builder(dataSet)
                .numLeaves(100).learningRate(0.1).numSplitIntervals(1000).minDataPerLeaf(2)
                .dataSamplingRate(1).featureSamplingRate(1).build();
        System.out.println(Arrays.toString(trainConfig.getActiveFeatures()));

        HMLGBTrainer trainer = new HMLGBTrainer(trainConfig,boosting);

        StopWatch stopWatch = new StopWatch();
        stopWatch.start();
        for (int round =0;round<10;round++){
            System.out.println("round="+round);
            trainer.iterate();
            System.out.println("accuracy="+Accuracy.accuracy(boosting,dataSet));

        }
        stopWatch.stop();
        System.out.println(stopWatch);
//        System.out.println(boosting);
        System.out.println("accuracy");
        System.out.println(Accuracy.accuracy(boosting,dataSet));
        System.out.println("macro-averaged:");
        System.out.println(new MacroAveragedMeasures(boosting,dataSet));
        boosting.serialize(new File(TMP,"/hmlgb/boosting.ser"));


    }

    static void test4_load() throws Exception{

        MultiLabelClfDataSet dataSet = TRECFormat.loadMultiLabelClfDataSet(new File(DATASETS,"spam/4labels/test.trec"),
                DataSetType.ML_CLF_DENSE,true);

        HMLGradientBoosting boosting = HMLGradientBoosting.deserialize(new File(TMP,"/hmlgb/boosting.ser"));
        System.out.println("accuracy="+Accuracy.accuracy(boosting,dataSet));
        System.out.println("macro-averaged:");
        System.out.println(new MacroAveragedMeasures(boosting,dataSet));

    }


    static void test5() throws Exception{
        MultiLabelClfDataSet dataSet = TRECFormat.loadMultiLabelClfDataSet(new File(DATASETS, "ohsumed/3/train.trec"), DataSetType.ML_CLF_SPARSE, true);
        MultiLabelClfDataSet testSet = TRECFormat.loadMultiLabelClfDataSet(new File(DATASETS, "ohsumed/3/test.trec"), DataSetType.ML_CLF_SPARSE, true);
        List<MultiLabel> assignments = DataSetUtil.gatherLabels(dataSet);
        HMLGradientBoosting boosting = new HMLGradientBoosting(dataSet.getNumClasses(),assignments);


        HMLGBConfig trainConfig = new HMLGBConfig.Builder(dataSet)
                .numLeaves(2).learningRate(0.1).numSplitIntervals(1000).minDataPerLeaf(2)
                .dataSamplingRate(1).featureSamplingRate(1).build();

        HMLGBTrainer trainer = new HMLGBTrainer(trainConfig,boosting);
        StopWatch stopWatch = new StopWatch();
        stopWatch.start();

        for (int round =0;round<100;round++){
            System.out.println("round="+round);
            trainer.iterate();
            System.out.println(stopWatch);
        }

        System.out.println("training accuracy="+ Accuracy.accuracy(boosting, dataSet));
        System.out.println("training overlap = "+ Overlap.overlap(boosting, dataSet));
        System.out.println("test accuracy="+ Accuracy.accuracy(boosting, testSet));
        System.out.println("test overlap = "+ Overlap.overlap(boosting,testSet));
    }

}