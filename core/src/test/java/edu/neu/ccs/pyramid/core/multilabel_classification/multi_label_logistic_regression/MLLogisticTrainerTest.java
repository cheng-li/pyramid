package edu.neu.ccs.pyramid.core.multilabel_classification.multi_label_logistic_regression;

import edu.neu.ccs.pyramid.core.configuration.Config;
import edu.neu.ccs.pyramid.core.dataset.*;
import edu.neu.ccs.pyramid.core.eval.Accuracy;
import edu.neu.ccs.pyramid.core.eval.Overlap;

import edu.neu.ccs.pyramid.core.optimization.LBFGS;
import org.apache.commons.lang3.time.StopWatch;

import java.io.File;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.stream.Collectors;

public class MLLogisticTrainerTest {
    private static final Config config = new Config("config/local.properties");
    private static final String DATASETS = config.getString("input.datasets");
    private static final String TMP = config.getString("output.tmp");

    public static void main(String[] args) throws Exception{
//        test2();
////        test5_train();
////        test5_test();
        test6();
    }


    private static void test1() throws Exception{
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

        MLLogisticRegression mlLogisticRegression = new MLLogisticRegression(dataSet.getNumClasses(),dataSet.getNumFeatures(),
                assignments);
        MLLogisticLoss function = new MLLogisticLoss(mlLogisticRegression,dataSet,10000);
        LBFGS lbfgs = new LBFGS(function);
        lbfgs.getTerminator().setRelativeEpsilon(0.01);
        lbfgs.setHistory(5);
        for (int i=0;i<100;i++){
            System.out.println(function.getValue());
//            System.out.println(Accuracy.accuracy(mlLogisticRegression,dataSet));
            lbfgs.iterate();

        }

        System.out.println(Accuracy.accuracy(mlLogisticRegression,dataSet));
        System.out.println(Overlap.overlap(mlLogisticRegression,dataSet));
    }

    private static void test2() throws Exception{
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


        MLLogisticTrainer trainer = MLLogisticTrainer.getBuilder().setGaussianPriorVariance(10000)
                .build();
        MLLogisticRegression mlLogisticRegression =trainer.train(dataSet,assignments);


        System.out.println(Accuracy.accuracy(mlLogisticRegression,dataSet));
        System.out.println(Overlap.overlap(mlLogisticRegression,dataSet));
    }

    /**
     *      * add a fake label in spam data set, if x=spam and x_0<0.1, also label it as 2
     * @throws Exception
     */
    private static void test3() throws Exception{
        ClfDataSet singleLabeldataSet = TRECFormat.loadClfDataSet(new File(DATASETS, "/spam/trec_data/train.trec"),
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



        MLLogisticRegression mlLogisticRegression = new MLLogisticRegression(dataSet.getNumClasses(),dataSet.getNumFeatures(),
                assignments);
        MLLogisticLoss function = new MLLogisticLoss(mlLogisticRegression,dataSet,10000);
        LBFGS lbfgs = new LBFGS(function);
        lbfgs.getTerminator().setRelativeEpsilon(0.01);
        lbfgs.setHistory(5);
        for (int i=0;i<1000;i++){
//            System.out.println(function.getValue());
            System.out.println(Accuracy.accuracy(mlLogisticRegression,dataSet));
            lbfgs.iterate();

        }

        System.out.println(Accuracy.accuracy(mlLogisticRegression,dataSet));
        System.out.println(Overlap.overlap(mlLogisticRegression,dataSet));
    }

    /**
     * 0 is the same as 2
     * @throws Exception
     */
    private static void test4() throws Exception{
        ClfDataSet singleLabeldataSet = TRECFormat.loadClfDataSet(new File(DATASETS, "/spam/trec_data/train.trec"),
                DataSetType.CLF_DENSE, true);
        int numDataPoints = singleLabeldataSet.getNumDataPoints();
        int numFeatures = singleLabeldataSet.getNumFeatures();
        MultiLabelClfDataSet dataSet = MLClfDataSetBuilder.getBuilder()
                .numDataPoints(numDataPoints).numFeatures(numFeatures)
                .numClasses(3).build();
        int[] labels = singleLabeldataSet.getLabels();
        for (int i=0;i<numDataPoints;i++){
            dataSet.addLabel(i,labels[i]);
            if (labels[i]==0 ){
                dataSet.addLabel(i,2);
            }
            for (int j=0;j<numFeatures;j++){
                double value = singleLabeldataSet.getRow(i).get(j);
                dataSet.setFeatureValue(i,j,value);
            }
        }

//        System.out.println(Arrays.toString(dataSet.getMultiLabels()));

        List<MultiLabel> assignments = new ArrayList<>();
//        assignments.add(new MultiLabel().addLabel(0));
        assignments.add(new MultiLabel().addLabel(1));
        assignments.add(new MultiLabel().addLabel(0).addLabel(2));


        MLLogisticTrainer trainer = MLLogisticTrainer.getBuilder().setGaussianPriorVariance(10000)
                .build();
        MLLogisticRegression mlLogisticRegression =trainer.train(dataSet,assignments);

        for (int i=0;i<dataSet.getNumDataPoints();i++){
            MultiLabel label = dataSet.getMultiLabels()[i];
            MultiLabel pred = mlLogisticRegression.predict(dataSet.getRow(i));
            if (!label.equals(pred)){
                System.out.println("---");
                System.out.println("label = "+label);
                System.out.println("prediction = "+pred);
                System.out.println(Arrays.toString(mlLogisticRegression.predictClassScores(dataSet.getRow(i))));
            }
        }

        System.out.println(Accuracy.accuracy(mlLogisticRegression,dataSet));
        System.out.println(Overlap.overlap(mlLogisticRegression,dataSet));
    }




    /**
     * ohsumed 20000
     * @throws Exception
     */
    private static void test5_train() throws Exception{
        MultiLabelClfDataSet dataSet = TRECFormat.loadMultiLabelClfDataSet(new File(DATASETS,"/ohsumed/unigrams/train.trec")
        ,DataSetType.ML_CLF_SPARSE,true);
        System.out.println(dataSet.getMetaInfo());
        List<MultiLabel> assignments = DataSetUtil.gatherMultiLabels(dataSet).stream()
                .collect(Collectors.toList());
        MLLogisticTrainer trainer = MLLogisticTrainer.getBuilder().setGaussianPriorVariance(1)
                .build();
        MLLogisticRegression mlLogisticRegression =trainer.train(dataSet,assignments);
        System.out.println("training accuracy = " +Accuracy.accuracy(mlLogisticRegression,dataSet));
        mlLogisticRegression.serialize(new File(TMP,"model"));

    }

    private static void test5_test() throws Exception{
        MLLogisticRegression mlLogisticRegression = MLLogisticRegression.deserialize(new File(TMP,"model"));

        MultiLabelClfDataSet testSet = TRECFormat.loadMultiLabelClfDataSet(new File(DATASETS,"/ohsumed/unigrams/test.trec")
                ,DataSetType.ML_CLF_SPARSE,true);
        System.out.println("test accuracy = " +Accuracy.accuracy(mlLogisticRegression,testSet));
        System.out.println("test overlap = " + Overlap.overlap(mlLogisticRegression,testSet));
    }

    static void test6() throws Exception{
        MultiLabelClfDataSet dataSet = TRECFormat.loadMultiLabelClfDataSet(new File(DATASETS, "ohsumed/3/train.trec"), DataSetType.ML_CLF_SPARSE, true);
        MultiLabelClfDataSet testSet = TRECFormat.loadMultiLabelClfDataSet(new File(DATASETS, "ohsumed/3/test.trec"), DataSetType.ML_CLF_SPARSE, true);
        List<MultiLabel> assignments = DataSetUtil.gatherMultiLabels(dataSet);
        MLLogisticTrainer trainer = MLLogisticTrainer.getBuilder().setGaussianPriorVariance(1)
                .build();
        StopWatch stopWatch = new StopWatch();
        stopWatch.start();
        MLLogisticRegression mlLogisticRegression =trainer.train(dataSet,assignments);
        System.out.println(stopWatch);


        System.out.println("training accuracy="+ Accuracy.accuracy(mlLogisticRegression, dataSet));
        System.out.println("training overlap = "+ Overlap.overlap(mlLogisticRegression, dataSet));
        System.out.println("test accuracy="+ Accuracy.accuracy(mlLogisticRegression, testSet));
        System.out.println("test overlap = "+ Overlap.overlap(mlLogisticRegression,testSet));
    }

}