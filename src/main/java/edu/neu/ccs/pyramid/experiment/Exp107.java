package edu.neu.ccs.pyramid.experiment;

import edu.neu.ccs.pyramid.dataset.DataSetType;
import edu.neu.ccs.pyramid.dataset.DataSetUtil;
import edu.neu.ccs.pyramid.dataset.RegDataSet;
import edu.neu.ccs.pyramid.dataset.TRECFormat;
import edu.neu.ccs.pyramid.eval.MSE;
import edu.neu.ccs.pyramid.optimization.LBFGS;
import edu.neu.ccs.pyramid.regression.probabilistic_regression_tree.ProbRegStump;
import edu.neu.ccs.pyramid.regression.probabilistic_regression_tree.ProbRegStumpTrainer;
import edu.neu.ccs.pyramid.regression.regression_tree.RegTreeConfig;
import edu.neu.ccs.pyramid.regression.regression_tree.RegTreeTrainer;
import edu.neu.ccs.pyramid.regression.regression_tree.RegressionTree;
import edu.neu.ccs.pyramid.util.Pair;

import java.io.File;
import java.util.ArrayList;
import java.util.List;
import java.util.stream.IntStream;

/**
 * hard regression stump vs soft regresson stump on real datasets
 * use the same split feature
 * Created by chengli on 5/25/15.
 */
public class Exp107 {

    public static void main(String[] args) throws Exception{
//        String data = "/Users/chengli/Dropbox/Shared/pyramid_shared/Datasets/housing/trec_format";

        String data = "/Users/chengli/Dropbox/Shared/pyramid_shared/Datasets/slice_location/trec_format";
        testDataSet(data);
    }



    static void testDataSet(String path) throws Exception{
        RegDataSet all = TRECFormat.loadRegDataSet(new File(path, "all.trec"), DataSetType.REG_DENSE,false);
        List<Double> hardTreeTrainPerf = new ArrayList<>();
        List<Double> hardTreeTestPerf = new ArrayList<>();
        List<Double> softTreeTrainPerf = new ArrayList<>();
        List<Double> softTreeTestPerf = new ArrayList<>();
        for (int run=0;run<1;run++){
            Pair<RegDataSet,RegDataSet> pair = DataSetUtil.splitToTrainValidation(all,0.8);
            RegDataSet trainSet = pair.getFirst();
            RegDataSet testSet = pair.getSecond();

            int[] activeFeatures = IntStream.range(0, trainSet.getNumFeatures()).toArray();
            int[] activeDataPoints = IntStream.range(0, trainSet.getNumDataPoints()).toArray();
            RegTreeConfig regTreeConfig = new RegTreeConfig();
            regTreeConfig.setActiveFeatures(activeFeatures);

            regTreeConfig.setMaxNumLeaves(2);
            regTreeConfig.setMinDataPerLeaf(5);
            regTreeConfig.setActiveDataPoints(activeDataPoints);

            regTreeConfig.setNumSplitIntervals(1000);
            RegressionTree hardTree = RegTreeTrainer.fit(regTreeConfig, trainSet);
            System.out.println("hard tree = "+hardTree);

            hardTreeTrainPerf.add(MSE.mse(hardTree,trainSet));
            hardTreeTestPerf.add(MSE.mse(hardTree,testSet));



            ProbRegStumpTrainer trainer = ProbRegStumpTrainer.getBuilder()
                    .setDataSet(trainSet)
                    .setLabels(trainSet.getLabels())
                    .setFeatureType(ProbRegStumpTrainer.FeatureType.ALL_FEATURES)
                    .setLossType(ProbRegStumpTrainer.LossType.SquaredLossOfExpectation)
                    .build();

            LBFGS lbfgs = trainer.getLbfgs();
            lbfgs.setCheckConvergence(false);
            lbfgs.setMaxIteration(100);
            ProbRegStump softTree = trainer.train();
            System.out.println("soft tree = "+softTree);

            softTreeTrainPerf.add(MSE.mse(softTree,trainSet));
            softTreeTestPerf.add(MSE.mse(softTree,testSet));

        }

        System.out.println("hard tree train performance "+hardTreeTrainPerf);
        System.out.println("hard tree test performance "+hardTreeTestPerf);
        System.out.println("soft tree train performance "+softTreeTrainPerf);
        System.out.println("soft tree test performance "+softTreeTestPerf);


    }
}
