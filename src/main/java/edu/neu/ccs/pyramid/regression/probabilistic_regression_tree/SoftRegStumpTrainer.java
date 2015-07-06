package edu.neu.ccs.pyramid.regression.probabilistic_regression_tree;

import edu.neu.ccs.pyramid.dataset.DataSet;
import edu.neu.ccs.pyramid.optimization.GradientDescent;
import edu.neu.ccs.pyramid.optimization.LBFGS;
import edu.neu.ccs.pyramid.optimization.Optimizable;
import edu.neu.ccs.pyramid.optimization.Optimizer;
import edu.neu.ccs.pyramid.regression.regression_tree.RegTreeConfig;
import edu.neu.ccs.pyramid.regression.regression_tree.RegTreeTrainer;
import edu.neu.ccs.pyramid.regression.regression_tree.RegressionTree;

import java.util.stream.IntStream;

/**
 * Created by chengli on 5/21/15.
 */
public class SoftRegStumpTrainer {
    private DataSet dataSet;
    private FeatureType featureType;
    private LossType lossType;
    private Optimizable.ByGradientValue loss;
    private Optimizer optimizer;

    public SoftRegStumpTrainer() {
    }

    public static Builder getBuilder(){
        return new Builder();
    }

    public Optimizer getOptimizer() {
        return optimizer;
    }

    /**
     * set parameters here
     * @return
     */


    public SoftRegStump train(){
        optimizer.optimize();

        SoftRegStump softRegStump = new SoftRegStump();
        softRegStump.lossType = lossType;

        switch (this.lossType) {
            case SquaredLossOfExpectation:
                SquaredLossOfExpectation squaredLossOfExpectation = (SquaredLossOfExpectation)loss;
                softRegStump.gatingFunction = new Sigmoid(squaredLossOfExpectation.getWeightsWithoutBias(), squaredLossOfExpectation.getBias());
                softRegStump.leftOutput = squaredLossOfExpectation.getLeftValue();
                softRegStump.rightOutput = squaredLossOfExpectation.getRightValue();

                break;

            case ExpectationOfSquaredLoss:
                ExpectationOfSquaredLoss expectationOfSquaredLoss = (ExpectationOfSquaredLoss)loss;
                softRegStump.gatingFunction = new Sigmoid(expectationOfSquaredLoss.getWeightsWithoutBias(), expectationOfSquaredLoss.getBias());
                softRegStump.leftOutput = expectationOfSquaredLoss.getLeftValue();
                softRegStump.rightOutput = expectationOfSquaredLoss.getRightValue();

                break;
        }

        softRegStump.featureList = dataSet.getFeatureList();

        return softRegStump;
    }

    public static enum FeatureType{
        ALL_FEATURES, FOLLOW_HARD_TREE_FEATURE
    }

    public static enum LossType{
        SquaredLossOfExpectation, ExpectationOfSquaredLoss
    }

    public static enum OptimizerType{
        GRADIENT_DESCENT, LBFGS
    }


    public static class Builder {
        private DataSet dataSet;
        private SoftRegStumpTrainer.FeatureType featureType;
        private SoftRegStumpTrainer.LossType lossType;
        private OptimizerType optimizerType=OptimizerType.LBFGS;
        private double[] labels;

        public Builder setDataSet(DataSet dataSet) {
            this.dataSet = dataSet;
            return this;
        }

        public Builder setFeatureType(SoftRegStumpTrainer.FeatureType featureType) {
            this.featureType = featureType;
            return this;
        }

        public Builder setLossType(SoftRegStumpTrainer.LossType lossType) {
            this.lossType = lossType;
            return this;
        }


        public Builder setLabels(double[] labels) {
            this.labels = labels;
            return this;
        }

        public Builder setOptimizerType(OptimizerType optimizerType) {
            this.optimizerType = optimizerType;
            return this;
        }

        public SoftRegStumpTrainer build() {
            RegressionTree hardTree = null;
            SoftRegStumpTrainer trainer = new SoftRegStumpTrainer();
            trainer.dataSet = dataSet;
            trainer.featureType = featureType;
            trainer.lossType = lossType;
            int[] activeFeatures = null;
            switch (featureType) {
                case ALL_FEATURES:
                    activeFeatures = IntStream.range(0, dataSet.getNumFeatures()).toArray();
                    break;
                case FOLLOW_HARD_TREE_FEATURE:
                    // train a hard tree first
                    RegTreeConfig regTreeConfig = new RegTreeConfig();

                    regTreeConfig.setMaxNumLeaves(2);
                    regTreeConfig.setMinDataPerLeaf(1);

                    regTreeConfig.setNumSplitIntervals(1000);
                    hardTree = RegTreeTrainer.fit(regTreeConfig, dataSet,labels);
                    int usedFeature = hardTree.getRoot().getFeatureIndex();
                    activeFeatures = new int[1];
                    activeFeatures[0] = usedFeature;
                    break;
            }

            switch (lossType) {
                //todo re-factor
                case SquaredLossOfExpectation:
                    trainer.loss = new SquaredLossOfExpectation(dataSet,labels, activeFeatures,hardTree);
                    break;
                case ExpectationOfSquaredLoss:
                    trainer.loss = new ExpectationOfSquaredLoss(dataSet,labels, activeFeatures);
                    break;
            }

            switch (optimizerType) {
                case GRADIENT_DESCENT:
                    trainer.optimizer = new GradientDescent(trainer.loss);
                    break;
                case LBFGS:
                    trainer.optimizer = new LBFGS(trainer.loss);
            }
            return trainer;
        }
    }
}
