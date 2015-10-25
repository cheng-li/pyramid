package edu.neu.ccs.pyramid.classification.logistic_regression;

import edu.neu.ccs.pyramid.dataset.DataSet;
import edu.neu.ccs.pyramid.dataset.GradientMatrix;
import edu.neu.ccs.pyramid.dataset.ProbabilityMatrix;
import edu.neu.ccs.pyramid.optimization.Optimizable;
import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.Vector;

import java.util.stream.IntStream;

/**
 * Created by Rainicy on 10/24/15.
 */
public class WeightedLogisticLoss implements Optimizable.ByGradientValue {
    private LogisticRegression logisticRegression;
    private DataSet dataSet;
    private double[] gammas;
    private double[][] targetDistributions;
    private double gaussianPriorVariance;
    private Vector empiricalCounts;
    private Vector predictedCounts;
    private Vector gradient;
    private int numParameters;
    private int numClasses;
    /**
     * numDataPoints by numClasses;
     */
    private ProbabilityMatrix probabilityMatrix;

    //todo the concept is not unified in logistic regression and gradient boosting

    /**
     * actually negative gradient
     *  y_ik - p_k(x_i)
     * numClasses by numDataPoints
     */
    private GradientMatrix gradientMatrix;
    private double value;
    private boolean isGradientCacheValid;
    private boolean isValueCacheValid;


    public WeightedLogisticLoss(LogisticRegression logisticRegression,
                          DataSet dataSet, double[] gammas ,double[][] targetDistributions,
                          double gaussianPriorVariance) {
        this.logisticRegression = logisticRegression;
        this.targetDistributions = targetDistributions;
        numParameters = logisticRegression.getWeights().totalSize();
        this.dataSet = dataSet;
        this.gammas = gammas;
        this.gaussianPriorVariance = gaussianPriorVariance;
        this.empiricalCounts = new DenseVector(numParameters);
        this.predictedCounts = new DenseVector(numParameters);
        this.numClasses = targetDistributions[0].length;
        this.probabilityMatrix = new ProbabilityMatrix(dataSet.getNumDataPoints(),numClasses);
        this.gradientMatrix = new GradientMatrix(dataSet.getNumDataPoints(),numClasses, GradientMatrix.Objective.MAXIMIZE);
        this.updateEmpricalCounts();
        this.isValueCacheValid=false;
        this.isGradientCacheValid=false;
    }



    public Vector getParameters(){
        return logisticRegression.getWeights().getAllWeights();
    }




    public void setParameters(Vector parameters) {
        this.logisticRegression.getWeights().setWeightVector(parameters);
        this.isValueCacheValid=false;
        this.isGradientCacheValid=false;

    }


    public double getValue() {
        if (isValueCacheValid){
            return this.value;
        }
        Vector parameters = getParameters();
        this.value =  logisticRegression.dataSetKLWeightedDivergence(dataSet, targetDistributions, gammas)
                + parameters.dot(parameters)/(2*gaussianPriorVariance);
        this.isValueCacheValid = true;
        return this.value;
    }



    public Vector getGradient(){
        if (isGradientCacheValid){
            return this.gradient;
        }
        updateClassProbMatrix();
        updatePredictedCounts();
        updateGradient();
        this.isGradientCacheValid = true;
        return this.gradient;
    }


    private void updateGradient(){
        Vector weights = this.logisticRegression.getWeights().getAllWeights();
        this.gradient = this.predictedCounts.minus(empiricalCounts).plus(weights.divide(gaussianPriorVariance));
    }

    private void updateEmpricalCounts(){
        IntStream.range(0, numParameters).parallel()
                .forEach(i -> this.empiricalCounts.set(i, calEmpricalCount(i)));
    }

    private void updatePredictedCounts(){
        IntStream.range(0,numParameters).parallel()
                .forEach(i -> this.predictedCounts.set(i, calPredictedCount(i)));
    }

    private double calEmpricalCount(int parameterIndex){
        int classIndex = logisticRegression.getWeights().getClassIndex(parameterIndex);
        int featureIndex = logisticRegression.getWeights().getFeatureIndex(parameterIndex);
        double count = 0;
        //bias
        if (featureIndex == -1){
            for (int i=0;i<dataSet.getNumDataPoints();i++){
                count += targetDistributions[i][classIndex]*gammas[i];
            }
        } else {
            Vector featureColumn = dataSet.getColumn(featureIndex);
            for (Vector.Element element: featureColumn.nonZeroes()){
                int dataPointIndex = element.index();
                double featureValue = element.get();
                //TODO added weighted
                count += featureValue*targetDistributions[dataPointIndex][classIndex]*gammas[dataPointIndex];
            }
        }
        return count;
    }

    private double calPredictedCount(int parameterIndex){
        int classIndex = logisticRegression.getWeights().getClassIndex(parameterIndex);
        int featureIndex = logisticRegression.getWeights().getFeatureIndex(parameterIndex);
        double count = 0;
        double[] probs = this.probabilityMatrix.getProbabilitiesForClass(classIndex);
        //bias
        if (featureIndex == -1){
            for (int i=0;i<dataSet.getNumDataPoints();i++){
                count += probs[i]*gammas[i];
            }
        } else {
            Vector featureColumn = dataSet.getColumn(featureIndex);
            for (Vector.Element element: featureColumn.nonZeroes()){
                int dataPointIndex = element.index();
                double featureValue = element.get();
                //TODO added weighted
                count += probs[dataPointIndex]*featureValue*gammas[dataPointIndex];
            }
        }
        return count;
    }

    private void updateClassProbs(int dataPointIndex){
        double[] probs = logisticRegression.predictClassProbs(dataSet.getRow(dataPointIndex));
        for (int k=0;k<numClasses;k++){
            this.probabilityMatrix.setProbability(dataPointIndex,k,probs[k]);
        }
    }

    private void updateClassProbMatrix(){
        IntStream.range(0,dataSet.getNumDataPoints()).parallel()
                .forEach(this::updateClassProbs);
    }

    private void updataDataGradient(int dataPointIndex){
        double[] classProbs = this.probabilityMatrix.getProbabilitiesForData(dataPointIndex);
        for (int k=0;k<numClasses;k++){
            this.gradientMatrix.setGradient(dataPointIndex,k,targetDistributions[dataPointIndex][k] - classProbs[k]);
        }
    }

    private void updateDataGradientMatrix(){
        IntStream.range(0,dataSet.getNumDataPoints()).parallel()
                .forEach(this::updataDataGradient);
    }


    public ProbabilityMatrix getProbabilityMatrix() {
        return probabilityMatrix;
    }

    public GradientMatrix getGradientMatrix() {
        return gradientMatrix;
    }


}
