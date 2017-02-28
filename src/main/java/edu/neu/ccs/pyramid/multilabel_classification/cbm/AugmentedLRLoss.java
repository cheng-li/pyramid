package edu.neu.ccs.pyramid.multilabel_classification.cbm;

import edu.neu.ccs.pyramid.dataset.DataSetUtil;
import edu.neu.ccs.pyramid.dataset.MultiLabelClfDataSet;
import edu.neu.ccs.pyramid.optimization.Optimizable;
import edu.neu.ccs.pyramid.util.MathUtil;
import edu.neu.ccs.pyramid.util.Vectors;
import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.Vector;

/**
 * Created by chengli on 2/28/17.
 */
public class AugmentedLRLoss implements Optimizable.ByGradientValue{
    private MultiLabelClfDataSet dataSet;
    // format [#data][#components]
    private double[][] gammas;
    private AugmentedLR augmentedLR;

    private int[] binaryLabels;
    private int numFeatures;
    private int numComponents;
    private int numData;

    private Vector empiricalCounts;
    private Vector predictedCounts;
    private Vector gradient;


    // size N*K*2
    // log probability of getting 0 and 1
    private double[][][] logProbs;

    // expected probability of getting 1
    // \sum_k \gamma_i^k p(y_i=1|x_i, z_i=k)
    //size = N
    private double[] expectedProbs;



    private double value;
    private boolean isGradientCacheValid;
    private boolean isValueCacheValid;
    private boolean isProbabilityCacheValid;
    private double priorGaussianVariance;

    private boolean regularizeAll;


    public AugmentedLRLoss(MultiLabelClfDataSet dataSet, int labelIndex, double[][] gammas,
                           AugmentedLR augmentedLR, double priorGaussianVariance, boolean regularizeAll) {
        this.dataSet = dataSet;
        this.gammas = gammas;
        this.augmentedLR = augmentedLR;
        this.priorGaussianVariance = priorGaussianVariance;
        this.regularizeAll = regularizeAll;
        this.binaryLabels = DataSetUtil.toBinaryLabels(dataSet.getMultiLabels(),labelIndex);
        this.numFeatures = dataSet.getNumFeatures();
        this.numComponents = augmentedLR.getNumComponents();
        this.empiricalCounts = new DenseVector(numFeatures+numComponents+1);
        this.predictedCounts = new DenseVector(numFeatures+numComponents+1);
        this.numData = dataSet.getNumDataPoints();
        this.logProbs = new double[numData][numComponents][2];
        this.expectedProbs = new double[numData];
        updateEmpiricalCounts();
        this.isGradientCacheValid = false;
        this.isValueCacheValid = false;
        this.isProbabilityCacheValid = false;
    }

    @Override
    public Vector getParameters() {
        return augmentedLR.getAllWeights();
    }

    @Override
    public void setParameters(Vector parameters) {
        augmentedLR.setWeights(parameters);
    }

    @Override
    public double getValue() {
        if (isValueCacheValid){
            return this.value;
        }
        double nll = computeNLL();
        this.value =  nll+penalty();
        this.isValueCacheValid = true;
        return this.value;
    }

    public Vector getGradient(){
        if (isGradientCacheValid){
            return this.gradient;
        }
        updateProbs();
        updatePredictedCounts();
        updateGradient();
        this.isGradientCacheValid = true;
        return this.gradient;
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
        for (int i=0;i<numData;i++){
            sum += ((double)binaryLabels[i])*gammas[i][k];
        }
        return sum;
    }

    private double calEmpiricalCountBias(){
        double sum = 0;
        for (int i=0;i<numData;i++){
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
        for (int i=0;i<numData;i++){
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
        for (int i=0;i<numData;i++){
            sum += Math.exp(logProbs[i][k][1])*gammas[i][k];
        }
        return sum;
    }

    private void updateExpectedProb(int i){
        double sum = 0;
        for (int k=0;k<numComponents;k++){
            sum += gammas[i][k]*Math.exp(logProbs[i][k][1]);
        }
        expectedProbs[i] = sum;
    }


    private void updateExpectedProbs(){
        for (int i=0;i<numData;i++){
            updateExpectedProb(i);
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

    private double penalty(){
        double square;
        if (regularizeAll){
            Vector weights = augmentedLR.getWeightsWithoutBias();
            square = Vectors.dot(weights, weights);
        } else {
            // only penalize feature weights
            Vector weights = augmentedLR.featureWeights();
            square = Vectors.dot(weights, weights);
        }
        return square/(2*priorGaussianVariance);
    }


    private Vector penaltyGradient(){
        Vector weightsVector = augmentedLR.getAllWeights();
        Vector penalty = new DenseVector(weightsVector.size());

        if (regularizeAll){
            for (int d=0;d<numFeatures+numComponents;d++){
                penalty.set(d, weightsVector.get(d)/priorGaussianVariance);
            }
        } else {
            for (int d=0;d<numFeatures;d++){
                penalty.set(d, weightsVector.get(d)/priorGaussianVariance);
            }
        }
        return penalty;
    }

    private void updateGradient(){
        this.gradient = this.predictedCounts.minus(empiricalCounts).plus(penaltyGradient());
    }

    private double computeNLL(){
        if (!isProbabilityCacheValid){
            updateProbs();
        }
        double sum = 0;
        for (int i=0;i<numData;i++){
            sum += computeNLL(i);
        }
        return sum;
    }

    private double computeNLL(int i){
        double sum = 0;
        int label = binaryLabels[i];
        for (int k=0;k<numComponents;k++){
            if (label==1){
                sum += logProbs[i][k][1];
            } else {
                sum += logProbs[i][k][0];
            }
        }
        return -1*sum;
    }


}
