package edu.neu.ccs.pyramid.multilabel_classification.bmm;

import edu.neu.ccs.pyramid.classification.logistic_regression.RidgeLogisticOptimizer;
import edu.neu.ccs.pyramid.dataset.MultiLabelClfDataSet;
import edu.neu.ccs.pyramid.optimization.*;

/**
 * Created by chengli on 10/7/15.
 */
public class BMMOptimizer {
    private BMMClassifier bmmClassifier;
    private MultiLabelClfDataSet dataSet;
    private Terminator terminator;
    double[][] gammas;
    // big variance means small regularization
    private double gaussianPriorVariance;

    public BMMOptimizer(BMMClassifier bmmClassifier, MultiLabelClfDataSet dataSet,
                        double gaussianPriorVariance) {
        this.bmmClassifier = bmmClassifier;
        this.dataSet = dataSet;
        this.gaussianPriorVariance = gaussianPriorVariance;
        this.terminator = new Terminator();
        this.gammas = new double[dataSet.getNumDataPoints()][bmmClassifier.numClusters];
    }

    public void optimize(){
        while (true){
            iterate();
            if (terminator.shouldTerminate()){
                break;
            }
        }
    }


    private void iterate(){
        eStep();
        mStep();
        this.terminator.add(getObjective());
    }

    private void eStep(){

    }


    private void mStep(){
        updateBernoullis();
        updateLogisticRegression();
    }


    private double getObjective(){
        return 0;
    }

    private void updateBernoullis(){

    }

    private void updateLogisticRegression(){
        RidgeLogisticOptimizer ridgeLogisticOptimizer = new RidgeLogisticOptimizer(this.gaussianPriorVariance);
        ridgeLogisticOptimizer.optimize(bmmClassifier.logisticRegression,dataSet,gammas);
    }
}
