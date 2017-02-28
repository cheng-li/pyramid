package edu.neu.ccs.pyramid.multilabel_classification.cbm;

import edu.neu.ccs.pyramid.dataset.MultiLabelClfDataSet;
import edu.neu.ccs.pyramid.optimization.Optimizable;
import edu.neu.ccs.pyramid.util.MathUtil;
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
    private int numFeatures;
    private int numComponents;

    private Vector empiricalCounts;
    private Vector predictedCounts;
    private Vector gradient;


    // size N*K
    private double[][] logProbs;

    // \sum_k \gamma_i^k p(y_i=1|x_i, z_i=k)
    //size = N
    private double[] expectedProbs;



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

    private void updateEmpiricalCounts(){
        for (int d=0;d<numFeatures;d++){
            double count = calEmpiricalCountFeatureWeight(d);
            empiricalCounts.set(d, count);
        }

        for (int k=0;k<numComponents;k++){
            double count = calEmpiricalCountComponentWeight(k);
            empiricalCounts.set(numFeatures + k, count);
        }

        empiricalCounts.set(numFeatures+numComponents, calEmpiricalCountBias());
    }

    private void updateProbs(){
        for (int i=0;i<dataSet.getNumDataPoints();i++){
            logProbs[i] = augmentedLR.logAugmentedProbs(dataSet.getRow(i));
        }

        updateExpectedProbs();

        this.isProbabilityCacheValid = true;
    }

    private double calPredictedCountFeatureWeight(int d){
        Vector featureColumn = dataSet.getColumn(d);
        double sum = 0;
        for (Vector.Element element: featureColumn.nonZeroes()){
            int dataIndex = element.index();
            double feature = element.get();
            sum += feature* expectedProbs[dataIndex];
        }
        return sum;
    }

    // k = component index
    private double calPredictedCountComponentWeight(int k){
        double sum = 0;
        for (int i=0;i<dataSet.getNumDataPoints();i++){
            sum += Math.exp(logProbs[i][k])*gammas[i][k];
        }
        return sum;
    }

    private void updateExpectedProb(int i){
        double sum = 0;
        for (int k=0;k<numComponents;k++){
            sum += gammas[i][k]*Math.exp(logProbs[i][k]);
        }
        expectedProbs[i] = sum;
    }


    private void updateExpectedProbs(){
        for (int i=0;i<dataSet.getNumDataPoints();i++){
            updateExpectedProbs();
        }
    }

    private double calPredictedCountBias(){
        return MathUtil.arraySum(expectedProbs);
    }

    private void updatePredictedCounts(){
        for (int d=0;d<numFeatures;d++){
            double count = calPredictedCountFeatureWeight(d);
            predictedCounts.set(d, count);
        }

        for (int k=0;k<numComponents;k++){
            double count = calPredictedCountComponentWeight(k);
            predictedCounts.set(numFeatures + k, count);
        }

        predictedCounts.set(numFeatures+numComponents, calPredictedCountBias());
    }




}
