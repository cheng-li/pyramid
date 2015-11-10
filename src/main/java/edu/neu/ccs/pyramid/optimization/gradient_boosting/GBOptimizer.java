package edu.neu.ccs.pyramid.optimization.gradient_boosting;

import edu.neu.ccs.pyramid.dataset.DataSet;
import edu.neu.ccs.pyramid.dataset.GradientMatrix;
import edu.neu.ccs.pyramid.dataset.ScoreMatrix;
import edu.neu.ccs.pyramid.regression.Regressor;
import edu.neu.ccs.pyramid.regression.RegressorFactory;
import edu.neu.ccs.pyramid.regression.regression_tree.RegressionTree;
import org.apache.mahout.math.Vector;

import java.util.Arrays;
import java.util.stream.IntStream;

/**
 * Created by chengli on 10/1/15.
 */
public abstract class GBOptimizer {
    protected ScoreMatrix scoreMatrix;
    /**
     * actually negative gradients, to be fit by the tree
     */
    protected GradientMatrix gradientMatrix;
    protected GradientBoosting boosting;
    protected RegressorFactory factory;
    protected DataSet dataSet;
    protected double[] weights;
    protected boolean isInitialized;
    protected double shrinkage = 1;


    protected GBOptimizer(GradientBoosting boosting, DataSet dataSet, double[] weights, RegressorFactory factory) {
        this.boosting = boosting;
        this.factory = factory;
        this.dataSet = dataSet;
        this.weights = weights;
    }

    protected GBOptimizer(GradientBoosting boosting, DataSet dataSet, RegressorFactory factory) {
        this(boosting,dataSet,defaultWeights(dataSet.getNumDataPoints()),factory);
    }

    /**
     * model specific initialization
     * should be called after constructor
     */
    public void initialize(){
        this.scoreMatrix = new ScoreMatrix(dataSet.getNumDataPoints(),boosting.getNumEnsembles());
        this.initStagedScores();
        initializeOthers();
        updateOthers();
        this.gradientMatrix = new GradientMatrix(dataSet.getNumDataPoints(),boosting.getNumEnsembles(), GradientMatrix.Objective.MAXIMIZE);
        updateGradientMatrix();
        this.isInitialized = true;
    }

    /**
     * e.g. probability matrix
     */
    protected abstract void initializeOthers();

    protected Regressor fitRegressor(int ensembleIndex){
        double[] gradients = this.gradientMatrix.getGradientsForClass(ensembleIndex);
        Regressor regressor = factory.fit(dataSet,gradients, weights);
        return regressor;
    }

    //todo make it more general
    protected void shrink(Regressor regressor){
        if (regressor instanceof RegressionTree){
            ((RegressionTree)regressor).shrink(shrinkage);
        }
    }

    protected void updateStagedScore(Regressor regressor, int ensembleIndex,
                                   int dataIndex){
        Vector vector = dataSet.getRow(dataIndex);
        double score = regressor.predict(vector);
        this.scoreMatrix.increment(dataIndex,ensembleIndex,score);
    }

    protected void updateStagedScores(Regressor regressor, int ensembleIndex){
        int numDataPoints = dataSet.getNumDataPoints();
        IntStream.range(0, numDataPoints).parallel()
                .forEach(dataIndex -> this.updateStagedScore(regressor,ensembleIndex,dataIndex));
    }

    public void iterate(){
        if (!isInitialized){
            throw new RuntimeException("GBOptimizer is not initialized");
        }
        for (int k=0;k<boosting.getNumEnsembles();k++){
            Regressor regressor = fitRegressor(k);
            shrink(regressor);
            boosting.getEnsemble(k).add(regressor);
            updateStagedScores(regressor,k);
        }
        updateOthers();
        updateGradientMatrix();
    }

    protected void initStagedScores(){
        for (int k=0;k<boosting.getNumEnsembles();k++){
            for (Regressor regressor: boosting.getEnsemble(k).getRegressors()){
                this.updateStagedScores(regressor,k);
            }
        }
    }

    /**
     * e.g. probability matrix
     */
    protected abstract void updateOthers();

    protected abstract void updateGradientMatrix();

    public void setShrinkage(double shrinkage) {
        this.shrinkage = shrinkage;
    }

    public RegressorFactory getRegressorFactory() {
        return factory;
    }

    protected static double[] defaultWeights(int numData){
        double[] weights = new double[numData];
        Arrays.fill(weights,1.0);
        return weights;
    }
}
