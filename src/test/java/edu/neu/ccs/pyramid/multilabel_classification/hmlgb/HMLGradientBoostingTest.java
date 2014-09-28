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
        spam_build();
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
        boosting.setTrainConfig(trainConfig);

        StopWatch stopWatch = new StopWatch();
        stopWatch.start();
        for (int round =0;round<200;round++){
            System.out.println("round="+round);
            boosting.boostOneRound();
//            System.out.println(Arrays.toString(boosting.getGradients(0)));
//            System.out.println(Arrays.toString(boosting.getGradients(1)));

        }
        stopWatch.stop();
        System.out.println(stopWatch);
        System.out.println(boosting);
        for (int i=0;i<numDataPoints;i++){
            FeatureRow featureRow = dataSet.getFeatureRow(i);
            System.out.println("label="+labels[i]);
//            System.out.println(boosting.calAssignmentScores(featureRow,assignments.get(0)));
//            System.out.println(boosting.calAssignmentScores(featureRow,assignments.get(1)));
//            System.out.println(boosting.predict(featureRow));
        }
        System.out.println("accuracy");
        System.out.println(Accuracy.accuracy(boosting,dataSet));

    }



}