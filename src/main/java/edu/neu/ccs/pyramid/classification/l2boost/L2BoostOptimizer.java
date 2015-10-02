package edu.neu.ccs.pyramid.classification.l2boost;

import edu.neu.ccs.pyramid.dataset.ClfDataSet;
import edu.neu.ccs.pyramid.dataset.DataSet;
import edu.neu.ccs.pyramid.dataset.ProbabilityMatrix;
import edu.neu.ccs.pyramid.optimization.gradient_boosting.GBOptimizer;
import edu.neu.ccs.pyramid.optimization.gradient_boosting.GradientBoosting;
import edu.neu.ccs.pyramid.regression.RegressorFactory;

import java.util.stream.IntStream;

/**
 * Created by chengli on 10/1/15.
 */
public class L2BoostOptimizer extends GBOptimizer {
    private L2Boost boosting;
    private ClfDataSet dataSet;
    private ProbabilityMatrix probabilityMatrix;

    public L2BoostOptimizer(L2Boost boosting, ClfDataSet dataSet, RegressorFactory factory) {
        super(boosting, dataSet, factory);
        this.boosting = boosting;
        this.dataSet = dataSet;
    }

    @Override
    protected void initializeOthers() {
        this.probabilityMatrix = new ProbabilityMatrix(dataSet.getNumDataPoints(),2);
    }

    @Override
    protected void updateOthers() {
        updateProbMatrix();
    }

    private void updateProbMatrix(){
        int numDataPoints = dataSet.getNumDataPoints();
        IntStream.range(0, numDataPoints).parallel()
                .forEach(this::updateProbability);
    }

    /**
     *
     * @param i data index
     */
    private void updateProbability(int i){
        // this is just a number at the moment
        double positiveScore = scoreMatrix.getScoresForData(i)[0];
        double[] scores = new double[2];
        scores[1] = positiveScore;
        double[] probs = boosting.predictClassProbs(scores);
        for (int k=0;k<2;k++){
            this.probabilityMatrix.setProbability(i,k,probs[k]);
        }
    }

    @Override
    protected void updateGradientMatrix() {
        int numDataPoints = dataSet.getNumDataPoints();
        IntStream.range(0, numDataPoints).parallel()
                .forEach(this::updateGradients);
    }

    /**
     * just for the positive class
     * @param dataPoint
     */
    private void updateGradients(int dataPoint){
        int label = dataSet.getLabels()[dataPoint];
        double[] probs = this.probabilityMatrix.getProbabilitiesForData(dataPoint);
        double gradient;
        if (label==1){
            gradient = 1-probs[1];
        } else {
            gradient = 0-probs[1];
        }
        this.gradientMatrix.setGradient(dataPoint,0,gradient);
    }
}
