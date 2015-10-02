package edu.neu.ccs.pyramid.optimization.gradient_boosting;

import edu.neu.ccs.pyramid.dataset.ClfDataSet;
import edu.neu.ccs.pyramid.dataset.DataSet;
import edu.neu.ccs.pyramid.dataset.GradientMatrix;
import edu.neu.ccs.pyramid.dataset.ScoreMatrix;
import edu.neu.ccs.pyramid.regression.Regressor;
import edu.neu.ccs.pyramid.regression.RegressorFactory;
import org.apache.mahout.math.Vector;

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

    public GBOptimizer(GradientBoosting boosting,  DataSet dataSet,RegressorFactory factory) {
        this.boosting = boosting;
        this.factory = factory;
        this.dataSet = dataSet;
        this.scoreMatrix = new ScoreMatrix(dataSet.getNumDataPoints(),boosting.getNumEnsembles());
        this.initStagedScores();
    }

    protected Regressor fitRegressor(int ensembleIndex){
        double[] gradients = this.gradientMatrix.getGradientsForClass(ensembleIndex);
        Regressor regressor = factory.fit(dataSet,gradients);
        return regressor;
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

    protected void iterate(){
        for (int k=0;k<boosting.getNumEnsembles();k++){
            Regressor regressor = fitRegressor(k);
            boosting.getEnsemble(k).add(regressor);
            updateStagedScores(regressor,k);
        }
    }

    protected void initStagedScores(){
        for (int k=0;k<boosting.getNumEnsembles();k++){
            for (Regressor regressor: boosting.getEnsemble(k).getRegressors()){
                this.updateStagedScores(regressor,k);
            }
        }
    }
}
