package edu.neu.ccs.pyramid.regression.probabilistic_regression_tree;

import edu.neu.ccs.pyramid.dataset.DataSet;
import edu.neu.ccs.pyramid.optimization.LBFGS;
import edu.neu.ccs.pyramid.regression.regression_tree.RegTreeConfig;
import edu.neu.ccs.pyramid.regression.regression_tree.RegTreeTrainer;
import edu.neu.ccs.pyramid.regression.regression_tree.RegressionTree;

import java.util.stream.IntStream;

/**
 * Created by chengli on 5/21/15.
 */
public class ProbRegStumpTrainer {
    private DataSet dataSet;
    private SquaredLossOfExpectation squaredLossOfExpectation;
    private LBFGS lbfgs;

    public ProbRegStumpTrainer(DataSet dataSet, double[] labels, FeatureType featureType) {
        this.dataSet = dataSet;
        int[] activeFeatures = null;
        switch (featureType) {
            case ALL_FEATURES:
                activeFeatures = IntStream.range(0, dataSet.getNumFeatures()).toArray();
                break;
            case FOLLOW_HARD_TREE_FEATURE:
                // train a hard tree first
                RegTreeConfig regTreeConfig = new RegTreeConfig();
                regTreeConfig.setActiveFeatures(IntStream.range(0, dataSet.getNumFeatures()).toArray());
                regTreeConfig.setMaxNumLeaves(2);
                regTreeConfig.setMinDataPerLeaf(1);
                regTreeConfig.setActiveDataPoints(IntStream.range(0, dataSet.getNumDataPoints()).toArray());
                regTreeConfig.setNumSplitIntervals(1000);
                RegressionTree hardTree = RegTreeTrainer.fit(regTreeConfig, dataSet,labels);
                int usedFeature = hardTree.getRoot().getFeatureIndex();
                activeFeatures = new int[1];
                activeFeatures[0] = usedFeature;
                break;
        }

        this.squaredLossOfExpectation = new SquaredLossOfExpectation(dataSet,labels, activeFeatures);
        this.lbfgs = new LBFGS(squaredLossOfExpectation);
    }

    /**
     * set parameters here
     * @return
     */
    public LBFGS getLbfgs() {
        return lbfgs;
    }

    public ProbRegStump train(){
        lbfgs.optimize();

//        GradientDescent gradientDescent = new GradientDescent(squaredLoss,1);
//        for (int i=0;i<100;i++){
//            gradientDescent.update();
//        }

        GatingFunction gatingFunction = new Sigmoid(squaredLossOfExpectation.getWeightsWithoutBias(), squaredLossOfExpectation.getBias());


        ProbRegStump probRegStump = new ProbRegStump();
        probRegStump.gatingFunction = gatingFunction;
        probRegStump.leftOutput = squaredLossOfExpectation.getLeftValue();
        probRegStump.rightOutput = squaredLossOfExpectation.getRightValue();
        probRegStump.featureList = dataSet.getFeatureList();

        return probRegStump;
    }

    public static enum FeatureType{
        ALL_FEATURES, FOLLOW_HARD_TREE_FEATURE
    }
}
