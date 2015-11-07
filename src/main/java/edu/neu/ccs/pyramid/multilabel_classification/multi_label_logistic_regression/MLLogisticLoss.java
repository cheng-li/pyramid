package edu.neu.ccs.pyramid.multilabel_classification.multi_label_logistic_regression;

import edu.neu.ccs.pyramid.dataset.MultiLabel;
import edu.neu.ccs.pyramid.dataset.MultiLabelClfDataSet;
import edu.neu.ccs.pyramid.optimization.Optimizable;
import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.Vector;

import java.util.stream.IntStream;

/**
 * Created by chengli on 12/23/14.
 */
public class MLLogisticLoss implements Optimizable.ByGradient, Optimizable.ByGradientValue{
    private MLLogisticRegression mlLogisticRegression;
    private MultiLabelClfDataSet dataSet;
    private double gaussianPriorVariance;
    private Vector empiricalCounts;
    private Vector predictedCounts;
    private Vector gradient;
    private int numParameters;
    /**
     * F_k(x), used to speed up training. classScoreMatrix.[i][k] = F_k(x_i)
     */
    private double[][] classScoreMatrix;
    /**
     * numDataPoints by numClasses;
     */
    private double[][] classProbMatrix;

    /**
     * [i][a]=prob of assignment a for x_i
     */
    private double[][] assignmentProbMatrix;

    private double[][] assignmentScoreMatrix;
    private double value;
    private boolean isGradientCacheValid;
    private boolean isValueCacheValid;

    public MLLogisticLoss(MLLogisticRegression mlLogisticRegression,
                        MultiLabelClfDataSet dataSet, double gaussianPriorVariance) {
        int numDataPoints = dataSet.getNumDataPoints();
        int numAssignments = mlLogisticRegression.getAssignments().size();
        int numClasses = dataSet.getNumClasses();
        this.mlLogisticRegression = mlLogisticRegression;
        numParameters = mlLogisticRegression.getWeights().totalSize();
        this.dataSet = dataSet;
        this.gaussianPriorVariance = gaussianPriorVariance;
        this.empiricalCounts = new DenseVector(numParameters);
        this.predictedCounts = new DenseVector(numParameters);
        this.classScoreMatrix = new double[numDataPoints][numClasses];
        this.classProbMatrix = new double[dataSet.getNumDataPoints()][dataSet.getNumClasses()];

        this.assignmentProbMatrix = new double[numDataPoints][numAssignments];
        this.assignmentScoreMatrix = new double[numDataPoints][numAssignments];
        this.updateEmpricalCounts();
        this.isValueCacheValid=false;
        this.isGradientCacheValid=false;
    }

    @Override
    public void setParallelism(boolean isParallel) {

    }

    @Override
    public boolean isParallel() {
        return false;
    }

    public Vector getParameters(){
        return mlLogisticRegression.getWeights().getAllWeights();
    }

    @Override
    public void setParameters(Vector parameters) {
        this.mlLogisticRegression.getWeights().setWeightVector(parameters);
        this.isValueCacheValid=false;
        this.isGradientCacheValid=false;
    }

    public double getValue(){
        if (isValueCacheValid){
            return this.value;
        }
        Vector parameters = getParameters();
        this.value =  -1*mlLogisticRegression.dataSetLogLikelihood(dataSet) + parameters.dot(parameters)/(2*gaussianPriorVariance);
        this.isValueCacheValid = true;
        return this.value;
    }


    public Vector getGradient(){
        if (isGradientCacheValid){
            return this.gradient;
        }
        updateClassScoreMatrix();
        updateAssignmentScoreMatrix();
        updateAssignmentProbMatrix();
        updateClassProbMatrix();
        updatePredictedCounts();
        updateGradient();
        this.isGradientCacheValid = true;
        return this.gradient;
    }


    private void updateGradient(){
        Vector weights = this.mlLogisticRegression.getWeights().getAllWeights();
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
        int classIndex = mlLogisticRegression.getWeights().getClassIndex(parameterIndex);
        MultiLabel[] labels = dataSet.getMultiLabels();
        int featureIndex = mlLogisticRegression.getWeights().getFeatureIndex(parameterIndex);
        double count = 0;
        //bias
        if (featureIndex == -1){
            for (int i=0;i<dataSet.getNumDataPoints();i++){
                if (labels[i].matchClass(classIndex)){
                    count +=1;
                }
            }
        } else {
            Vector featureColumn = dataSet.getColumn(featureIndex);
            for (Vector.Element element: featureColumn.nonZeroes()){
                int dataPointIndex = element.index();
                double featureValue = element.get();
                MultiLabel label = labels[dataPointIndex];
                if (label.matchClass(classIndex)){
                    count += featureValue;
                }
            }
        }
        return count;
    }

    private double calPredictedCount(int parameterIndex){
        int classIndex = mlLogisticRegression.getWeights().getClassIndex(parameterIndex);
        int featureIndex = mlLogisticRegression.getWeights().getFeatureIndex(parameterIndex);
        double count = 0;
        //bias
        if (featureIndex == -1){
            for (int i=0;i<dataSet.getNumDataPoints();i++){
                count += this.classProbMatrix[i][classIndex];
            }
        } else {
            Vector featureColumn = dataSet.getColumn(featureIndex);
            for (Vector.Element element: featureColumn.nonZeroes()){
                int dataPointIndex = element.index();
                double featureValue = element.get();
                count += this.classProbMatrix[dataPointIndex][classIndex] * featureValue;
            }
        }
        return count;
    }


    public double[] getClassProbs(int dataPointIndex){
        return classProbMatrix[dataPointIndex];
    }

    private void updateClassScoreMatrix(){
        IntStream.range(0,dataSet.getNumDataPoints()).parallel()
                .forEach(i-> classScoreMatrix[i] = mlLogisticRegression.predictClassScores(dataSet.getRow(i)));
    }

    private void updateAssignmentScoreMatrix(){
        IntStream.range(0,dataSet.getNumDataPoints()).parallel()
                .forEach(i -> assignmentScoreMatrix[i] = mlLogisticRegression.calAssignmentScores(classScoreMatrix[i]));
    }

    private void updateAssignmentProbMatrix(){
        IntStream.range(0,dataSet.getNumDataPoints()).parallel()
                .forEach(i -> assignmentProbMatrix[i] = mlLogisticRegression.calAssignmentProbs(assignmentScoreMatrix[i]));
    }

    private void updateClassProbMatrix(){
        IntStream.range(0,dataSet.getNumDataPoints()).parallel()
                .forEach(i -> classProbMatrix[i] = mlLogisticRegression.calClassProbs(assignmentProbMatrix[i]));
    }



}
