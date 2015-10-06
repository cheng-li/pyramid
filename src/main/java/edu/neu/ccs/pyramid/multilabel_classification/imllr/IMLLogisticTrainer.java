package edu.neu.ccs.pyramid.multilabel_classification.imllr;

import edu.neu.ccs.pyramid.dataset.MultiLabel;
import edu.neu.ccs.pyramid.dataset.MultiLabelClfDataSet;
import edu.neu.ccs.pyramid.optimization.LBFGS;

import java.util.List;

/**
 * Created by chengli on 5/15/15.
 */
public class IMLLogisticTrainer {
    private double gaussianPriorVariance = 1;
    private double epsilon = 1;
    private int history = 5;


    public static Builder getBuilder(){
        return new Builder();
    }

    public IMLLogisticRegression train(MultiLabelClfDataSet dataset, List<MultiLabel> assignments){

        IMLLogisticRegression IMLLogisticRegression = new IMLLogisticRegression(dataset.getNumClasses(),dataset.getNumFeatures(),
                assignments);
        IMLLogisticRegression.setFeatureList(dataset.getFeatureList());
        IMLLogisticRegression.setLabelTranslator(dataset.getLabelTranslator());

        IMLLogisticLoss function = new IMLLogisticLoss(IMLLogisticRegression,dataset,gaussianPriorVariance);
        LBFGS lbfgs = new LBFGS(function);
        lbfgs.getTerminator().setRelativeEpsilon(epsilon);
        lbfgs.setHistory(history);
        lbfgs.optimize();
        return IMLLogisticRegression;
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

        public IMLLogisticTrainer build(){
            IMLLogisticTrainer trainer = new IMLLogisticTrainer();
            trainer.gaussianPriorVariance = this.gaussianPriorVariance;
            trainer.epsilon = this.epsilon;
            trainer.history = this.history;
            return trainer;
        }
    }
}
