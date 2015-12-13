package edu.neu.ccs.pyramid.multilabel_classification.crf;

import edu.neu.ccs.pyramid.dataset.DataSet;
import edu.neu.ccs.pyramid.optimization.Optimizable;
import org.apache.mahout.math.Vector;

/**
 * Created by Rainicy on 12/13/15.
 */
public class CRFLoss implements Optimizable.ByGradientValue {
    private BinaryCRF binaryCRF;
    private DataSet dataSet;
    private double gaussianPriorVariance;
    /**
     * 
     */
    private Vector predictedCounts;


    private boolean isParallel = false;
    private boolean isGradientCacheValid = false;
    private boolean isValueCacheValid = false;


    public CRFLoss (BinaryCRF binaryCRF, DataSet dataSet, double gaussianPriorVariance) {
        this.binaryCRF = binaryCRF;
        this.dataSet = dataSet;
    }

    /**
     * gradient of log likelihood?
     * @return
     */
    @Override
    public Vector getGradient() {
        return null;
    }

    /**
     * TODO: log-likelihood?
     * @return
     */
    @Override
    public double getValue() {
        return 0;
    }

    @Override
    public Vector getParameters() {
        return binaryCRF.getWeights().getAllWeights();
    }

    @Override
    public void setParameters(Vector parameters) {
        this.binaryCRF.getWeights().setWeightVector(parameters);
        this.isValueCacheValid = false;
        this.isGradientCacheValid = false;
    }

    @Override
    public void setParallelism(boolean isParallel) {
        this.isParallel = isParallel;
    }

    @Override
    public boolean isParallel() {
        return this.isParallel;
    }
}
