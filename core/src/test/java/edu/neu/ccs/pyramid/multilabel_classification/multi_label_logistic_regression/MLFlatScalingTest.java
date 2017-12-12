package edu.neu.ccs.pyramid.multilabel_classification.multi_label_logistic_regression;

import edu.neu.ccs.pyramid.configuration.Config;
import edu.neu.ccs.pyramid.dataset.*;
import edu.neu.ccs.pyramid.multilabel_classification.MLFlatScaling;
import edu.neu.ccs.pyramid.multilabel_classification.imlgb.IMLGBConfig;
import edu.neu.ccs.pyramid.multilabel_classification.imlgb.IMLGBTrainer;
import edu.neu.ccs.pyramid.multilabel_classification.imlgb.IMLGradientBoosting;
import org.apache.commons.lang3.time.StopWatch;

import java.io.File;
import java.util.Arrays;
import java.util.List;

public class MLFlatScalingTest {
    private static final Config config = new Config("config/local.properties");
    private static final String DATASETS = config.getString("input.datasets");
    private static final String TMP = config.getString("output.tmp");

    public static void main(String[] args) throws Exception{
        test1();

    }

    private static void test1() throws Exception{
        MultiLabelClfDataSet dataSet = TRECFormat.loadMultiLabelClfDataSet(new File(DATASETS, "ohsumed/3/train.trec"), DataSetType.ML_CLF_SPARSE, true);
        IMLGradientBoosting boosting = new IMLGradientBoosting(dataSet.getNumClasses());
        List<MultiLabel> assignments = DataSetUtil.gatherMultiLabels(dataSet);
        boosting.setAssignments(assignments);

        IMLGBConfig trainConfig = new IMLGBConfig.Builder(dataSet)
                .numLeaves(2).learningRate(0.1).numSplitIntervals(1000).minDataPerLeaf(2)
                .build();

        IMLGBTrainer trainer = new IMLGBTrainer(trainConfig,boosting);
        StopWatch stopWatch = new StopWatch();
        stopWatch.start();

        for (int round =0;round<100;round++){
            System.out.println("round="+round);
            trainer.iterate();
            System.out.println(stopWatch);
        }

        MLFlatScaling scaling = new MLFlatScaling(dataSet,boosting);
        for (int i=0;i<10;i++){
            System.out.println(Arrays.toString(boosting.predictClassScores(dataSet.getRow(i))));
            System.out.println(Arrays.toString(boosting.predictClassProbs(dataSet.getRow(i))));
            System.out.println(Arrays.toString(scaling.predictClassProbs(dataSet.getRow(i))));
            System.out.println("======================");
        }
    }

}