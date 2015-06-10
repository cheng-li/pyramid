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
public class ProbRegStumpTrainer {
    private DataSet dataSet;
    private FeatureType featureType;
    private LossType lossType;
    private Optimizable.ByGradientValue loss;
    private Optimizer optimizer;

    public ProbRegStumpTrainer() {
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


    public ProbRegStump train(){
        optimizer.optimize();

        ProbRegStump probRegStump = new ProbRegStump();
        probRegStump.lossType = lossType;

        switch (this.lossType) {
            case SquaredLossOfExpectation:
                SquaredLossOfExpectation squaredLossOfExpectation = (SquaredLossOfExpectation)loss;
                probRegStump.gatingFunction = new Sigmoid(squaredLossOfExpectation.getWeightsWithoutBias(), squaredLossOfExpectation.getBias());
                probRegStump.leftOutput = squaredLossOfExpectation.getLeftValue();
                probRegStump.rightOutput = squaredLossOfExpectation.getRightValue();

                break;

            case ExpectationOfSquaredLoss:
                ExpectationOfSquaredLoss expectationOfSquaredLoss = (ExpectationOfSquaredLoss)loss;
                probRegStump.gatingFunction = new Sigmoid(expectationOfSquaredLoss.getWeightsWithoutBias(), expectationOfSquaredLoss.getBias());
                probRegStump.leftOutput = expectationOfSquaredLoss.getLeftValue();
                probRegStump.rightOutput = expectationOfSquaredLoss.getRightValue();

                break;
        }

        probRegStump.featureList = dataSet.getFeatureList();

        return probRegStump;
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
        private ProbRegStumpTrainer.FeatureType featureType;
        private ProbRegStumpTrainer.LossType lossType;
        private OptimizerType optimizerType=OptimizerType.LBFGS;
        private double[] labels;

        public Builder setDataSet(DataSet dataSet) {
            this.dataSet = dataSet;
            return this;
        }

        public Builder setFeatureType(ProbRegStumpTrainer.FeatureType featureType) {
            this.featureType = featureType;
            return this;
        }

        public Builder setLossType(ProbRegStumpTrainer.LossType lossType) {
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

        public ProbRegStumpTrainer build() {
            RegressionTree hardTree = null;
            ProbRegStumpTrainer trainer = new ProbRegStumpTrainer();
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
                    regTreeConfig.setActiveFeatures(IntStream.range(0, dataSet.getNumFeatures()).toArray());
                    regTreeConfig.setMaxNumLeaves(2);
                    regTreeConfig.setMinDataPerLeaf(5);
                    regTreeConfig.setActiveDataPoints(IntStream.range(0, dataSet.getNumDataPoints()).toArray());
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
