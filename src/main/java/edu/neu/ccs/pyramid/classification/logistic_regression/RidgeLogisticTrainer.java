package edu.neu.ccs.pyramid.classification.logistic_regression;

import edu.neu.ccs.pyramid.dataset.ClfDataSet;

import edu.neu.ccs.pyramid.optimization.LBFGS;


/**
 * Created by chengli on 11/28/14.
 */
public class RidgeLogisticTrainer {
    private double gaussianPriorVariance = 1;
    private double epsilon = 1;
    private int history = 5;


    public static Builder getBuilder(){
        return new Builder();
    }

    public LogisticRegression train(ClfDataSet clfDataSet){

        LogisticRegression logisticRegression = new LogisticRegression(clfDataSet.getNumClasses(),clfDataSet.getNumFeatures());
        logisticRegression.setFeatureList(clfDataSet.getFeatureList());
        logisticRegression.setFeatureExtraction(false);
        LogisticLoss function = new LogisticLoss(logisticRegression,clfDataSet,gaussianPriorVariance);
        LBFGS lbfgs = new LBFGS(function);
        lbfgs.setEpsilon(epsilon);
        lbfgs.setHistory(history);
        lbfgs.optimize();
        return logisticRegression;
    }

    public static class Builder {
        private double gaussianPriorVariance = 1;
        private double epsilon = 1;
        private int history = 5;

        public Builder setGaussianPriorVariance(double gaussianPriorVariance) {
            this.gaussianPriorVariance = gaussianPriorVariance;
            return this;
        }

        public Builder setEpsilon(double epsilon) {
            this.epsilon = epsilon;
            return this;
        }

        public Builder setHistory(int history) {
            this.history = history;
            return this;
        }

        public RidgeLogisticTrainer build(){
            RidgeLogisticTrainer trainer = new RidgeLogisticTrainer();
            trainer.gaussianPriorVariance = this.gaussianPriorVariance;
            trainer.epsilon = this.epsilon;
            trainer.history = this.history;
            return trainer;
        }
    }



}
