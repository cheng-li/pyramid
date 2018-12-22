package edu.neu.ccs.pyramid.calibration;

import edu.neu.ccs.pyramid.dataset.RegDataSet;
import edu.neu.ccs.pyramid.multilabel_classification.cbm.CBM;
import edu.neu.ccs.pyramid.regression.least_squares_boost.LSBoost;
import edu.neu.ccs.pyramid.regression.least_squares_boost.LSBoostOptimizer;
import edu.neu.ccs.pyramid.regression.ls_logistic_boost.LSLogisticBoost;
import edu.neu.ccs.pyramid.regression.ls_logistic_boost.LSLogisticBoostOptimizer;
import edu.neu.ccs.pyramid.regression.regression_tree.RegTreeConfig;
import edu.neu.ccs.pyramid.regression.regression_tree.RegTreeFactory;

public class RerankerTrainer {
    private int numLeaves;
    private int numIterations;
    private boolean monotonic;
    private int numCandidates;
    private double shrinkage;
    private int minDataPerLeaf;


    public Reranker train(RegDataSet regDataSet, CBM cbm, PredictionVectorizer predictionVectorizer){
        LSBoost lsBoost = new LSBoost();

        RegTreeConfig regTreeConfig = new RegTreeConfig().setMaxNumLeaves(numLeaves).setMinDataPerLeaf(minDataPerLeaf);
        RegTreeFactory regTreeFactory = new RegTreeFactory(regTreeConfig);
        LSBoostOptimizer optimizer = new LSBoostOptimizer(lsBoost, regDataSet, regTreeFactory);
        if (monotonic){
            optimizer.setMonotonicity(predictionVectorizer.getMonotonicityConstraints(cbm.getNumClasses()));
        }
        optimizer.setShrinkage(shrinkage);
        optimizer.initialize();

        for (int i=1;i<=numIterations;i++){
            optimizer.iterate();
        }

        return new Reranker(lsBoost, cbm, numCandidates,predictionVectorizer);
    }


    public Reranker trainWithSigmoid(RegDataSet regDataSet, CBM cbm, PredictionVectorizer predictionVectorizer){
        LSLogisticBoost lsLogisticBoost = new LSLogisticBoost();

        RegTreeConfig regTreeConfig = new RegTreeConfig().setMaxNumLeaves(numLeaves).setMinDataPerLeaf(minDataPerLeaf);
        RegTreeFactory regTreeFactory = new RegTreeFactory(regTreeConfig);
        LSLogisticBoostOptimizer optimizer = new LSLogisticBoostOptimizer(lsLogisticBoost, regDataSet, regTreeFactory);
        if (monotonic){
            optimizer.setMonotonicity(predictionVectorizer.getMonotonicityConstraints(cbm.getNumClasses()));
        }
        optimizer.setShrinkage(shrinkage);
        optimizer.initialize();

        for (int i=1;i<=numIterations;i++){
            optimizer.iterate();
        }

        return new Reranker(lsLogisticBoost, cbm, numCandidates,predictionVectorizer);
    }


    private RerankerTrainer(Builder builder) {
        numLeaves = builder.numLeaves;
        numIterations = builder.numIterations;
        monotonic = builder.monotonic;
        numCandidates = builder.numCandidates;
        shrinkage = builder.shrinkage;
        minDataPerLeaf = builder.minDataPerLeaf;

    }

    public static Builder newBuilder() {
        return new Builder();
    }


    public static final class Builder {
        private int numLeaves = 10;
        private int numIterations = 100;
        private boolean monotonic = true;
        private int numCandidates = 50;
        private double shrinkage = 0.1;
        private int minDataPerLeaf=5;

        private Builder() {
        }

        public Builder numLeaves(int val) {
            numLeaves = val;
            return this;
        }

        public Builder numIterations(int val) {
            numIterations = val;
            return this;
        }

        public Builder monotonic(boolean val) {
            monotonic = val;
            return this;
        }

        public Builder numCandidates(int val){
            numCandidates = val;
            return this;
        }

        public Builder shrinkage(double val){
            shrinkage = val;
            return this;
        }

        public Builder minDataPerLeaf(int val){
            minDataPerLeaf = val;
            return this;
        }

        public RerankerTrainer build() {
            return new RerankerTrainer(this);
        }
    }
}
