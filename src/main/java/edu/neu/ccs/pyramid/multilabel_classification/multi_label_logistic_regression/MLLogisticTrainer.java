package edu.neu.ccs.pyramid.multilabel_classification.multi_label_logistic_regression;

import edu.neu.ccs.pyramid.dataset.MultiLabel;
import edu.neu.ccs.pyramid.dataset.MultiLabelClfDataSet;
import edu.neu.ccs.pyramid.optimization.LBFGS;

import java.util.List;

/**
 * Created by chengli on 12/23/14.
 */
public class MLLogisticTrainer {
    private double gaussianPriorVariance = 1;
    private double epsilon = 1;
    private int history = 5;


    public static Builder getBuilder(){
        return new Builder();
    }

    public MLLogisticRegression train(MultiLabelClfDataSet dataset, List<MultiLabel> assignments){

        MLLogisticRegression mlLogisticRegression = new MLLogisticRegression(dataset.getNumClasses(),dataset.getNumFeatures(),
                assignments);
        mlLogisticRegression.setFeatureList(dataset.getFeatureList());
        mlLogisticRegression.setLabelTranslator(dataset.getLabelTranslator());
        mlLogisticRegression.setFeatureExtraction(false);
        MLLogisticLoss function = new MLLogisticLoss(mlLogisticRegression,dataset,gaussianPriorVariance);
        LBFGS lbfgs = new LBFGS(function);
        lbfgs.setEpsilon(epsilon);
        lbfgs.setHistory(history);
        lbfgs.optimize();
        return mlLogisticRegression;
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

        public MLLogisticTrainer build(){
            MLLogisticTrainer trainer = new MLLogisticTrainer();
            trainer.gaussianPriorVariance = this.gaussianPriorVariance;
            trainer.epsilon = this.epsilon;
            trainer.history = this.history;
            return trainer;
        }
    }
}
