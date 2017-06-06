package edu.neu.ccs.pyramid.core.multilabel_classification.multi_label_logistic_regression;

import edu.neu.ccs.pyramid.core.dataset.DataSetUtil;
import edu.neu.ccs.pyramid.core.dataset.MultiLabel;
import edu.neu.ccs.pyramid.core.dataset.MultiLabelClfDataSet;
import edu.neu.ccs.pyramid.core.optimization.LBFGS;

import java.util.List;
import java.util.stream.Collectors;

/**
 * Created by chengli on 12/23/14.
 */
public class MLLogisticTrainer {
    private double gaussianPriorVariance = 1;



    public static Builder getBuilder(){
        return new Builder();
    }

    public MLLogisticRegression train(MultiLabelClfDataSet dataset){
        List<MultiLabel> assignments = DataSetUtil.gatherMultiLabels(dataset).stream().collect(Collectors.toList());
        return this.train(dataset,assignments);
    }

    public MLLogisticRegression train(MultiLabelClfDataSet dataset, List<MultiLabel> assignments){

        MLLogisticRegression mlLogisticRegression = new MLLogisticRegression(dataset.getNumClasses(),dataset.getNumFeatures(),
                assignments);
        mlLogisticRegression.setFeatureList(dataset.getFeatureList());
        mlLogisticRegression.setLabelTranslator(dataset.getLabelTranslator());
        mlLogisticRegression.setFeatureExtraction(false);
        MLLogisticLoss function = new MLLogisticLoss(mlLogisticRegression,dataset,gaussianPriorVariance);
        LBFGS lbfgs = new LBFGS(function);
        lbfgs.optimize();
        return mlLogisticRegression;
    }

    public static class Builder {
        private double gaussianPriorVariance = 1;

        public Builder setGaussianPriorVariance(double gaussianPriorVariance) {
            this.gaussianPriorVariance = gaussianPriorVariance;
            return this;
        }



        public MLLogisticTrainer build(){
            MLLogisticTrainer trainer = new MLLogisticTrainer();
            trainer.gaussianPriorVariance = this.gaussianPriorVariance;
            return trainer;
        }
    }
}
