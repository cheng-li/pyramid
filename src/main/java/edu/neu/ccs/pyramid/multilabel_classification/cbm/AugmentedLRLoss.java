package edu.neu.ccs.pyramid.multilabel_classification.cbm;

import edu.neu.ccs.pyramid.dataset.MultiLabelClfDataSet;
import edu.neu.ccs.pyramid.optimization.Optimizable;
import org.apache.mahout.math.Vector;

/**
 * Created by chengli on 2/28/17.
 */
public class AugmentedLRLoss implements Optimizable.ByGradientValue{
    private MultiLabelClfDataSet dataSet;
    private int labelIndex;
    // format [#data][#components]
    private double[][] gammas;
    private AugmentedLR augmentedLR;

    private int[] binaryLabels;

    private Vector empiricalCounts;
    private Vector predictedCounts;
    private Vector gradient;


    private double[][] logProbabilityMatrixKByN;
    private double value;
    private boolean isGradientCacheValid;
    private boolean isValueCacheValid;
    private boolean isProbabilityCacheValid;
    private boolean isParallel = false;
    private double priorGaussianVariance;



    @Override
    public Vector getParameters() {
        return augmentedLR.getWeights();
    }

    @Override
    public void setParameters(Vector parameters) {
        augmentedLR.setWeights(parameters);
    }

    @Override
    public double getValue() {
        return 0;
    }

    @Override
    public Vector getGradient() {
        return null;
    }

    // d = feature index
    private double calEmpiricalCountFeatureWeight(int d){
        Vector featureColumn = dataSet.getColumn(d);
        double sum = 0;
        for (Vector.Element element: featureColumn.nonZeroes()){
            int dataIndex = element.index();
            double feature = element.get();
            if (binaryLabels[dataIndex]==1){
                sum += feature;
            }
        }
        return sum;
    }

    // k = component index
    private double calEmpiricalCountComponentWeight(int k){
        double sum = 0;
        for (int i=0;i<dataSet.getNumDataPoints();i++){
            sum += ((double)binaryLabels[i])*gammas[i][k];
        }
        return sum;
    }

    private double calEmpiricalCountBias(){
        double sum = 0;
        for (int i=0;i<dataSet.getNumDataPoints();i++){
            sum += binaryLabels[i];
        }
        return sum;
    }

}
