package edu.neu.ccs.pyramid.classification.logistic_regression;

import edu.neu.ccs.pyramid.dataset.ClfDataSet;
import edu.neu.ccs.pyramid.optimization.Optimizable;
import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.Vector;

import java.util.stream.IntStream;


/**
 * logistic loss, l2 regularized
 * to be minimized
 * references:
 * Maxent Models, Conditional Estimation, and Optimization, without the Magic,Dan Klein and Chris Manning
 * Conditional Random Fields, Rahul Gupta
 * Created by chengli on 12/7/14.
 */
public class LogisticLoss implements Optimizable.ByGradient, Optimizable.ByGradientValue{
    private LogisticRegression logisticRegression;
    private ClfDataSet dataSet;
    private double gaussianPriorVariance;
    private Vector empiricalCounts;
    private Vector predictedCounts;
    private Vector gradient;
    private int numParameters;
    /**
     * numDataPoints by numClasses;
     */
    private double[][] classProbMatrix;
    /**
     * p_k(x_i) - y_ik
     * numClasses by numDataPoints
     */
    private double[][] dataGradientMatrix;

    public LogisticLoss(LogisticRegression logisticRegression,
                        ClfDataSet dataSet, double gaussianPriorVariance) {
        this.logisticRegression = logisticRegression;
        numParameters = logisticRegression.getWeights().totalSize();
        this.dataSet = dataSet;
        this.gaussianPriorVariance = gaussianPriorVariance;
        this.empiricalCounts = new DenseVector(numParameters);
        this.predictedCounts = new DenseVector(numParameters);
        this.classProbMatrix = new double[dataSet.getNumDataPoints()][dataSet.getNumClasses()];
        this.dataGradientMatrix = new double[dataSet.getNumClasses()][dataSet.getNumDataPoints()];
        this.updateEmpricalCounts();
        this.refresh();
    }

    public Vector getParameters(){
        return logisticRegression.getWeights().getAllWeights();
    }

    public void refresh(){
        if (logisticRegression.featureExtraction()){
            updateEmpricalCounts();
        }
        updateClassProbMatrix();
        updatePredictedCounts();
        updateGradient();
        if (logisticRegression.featureExtraction()){
            updateDataGradientMatrix();
        }
    }

    @Override
    public double getValue(Vector parameters) {
        LogisticRegression tmpFunction = new LogisticRegression(this.logisticRegression.getNumClasses(),
                this.logisticRegression.getNumFeatures(),parameters);
        return -tmpFunction.dataSetLogLikelihood(dataSet) + parameters.norm(2)/(2*gaussianPriorVariance);
    }

    public Vector getGradient(){
        return this.gradient;
    }


    private void updateGradient(){
        Vector weights = this.logisticRegression.getWeights().getAllWeights();
        this.gradient = this.predictedCounts.minus(empiricalCounts).plus(weights.divide(gaussianPriorVariance));
    }

    private void updateEmpricalCounts(){
        IntStream.range(0,numParameters).parallel()
                .forEach(i -> this.empiricalCounts.set(i, calEmpricalCount(i)));
    }

    private void updatePredictedCounts(){
        IntStream.range(0,numParameters).parallel()
                .forEach(i -> this.predictedCounts.set(i, calPredictedCount(i)));
    }

    private double calEmpricalCount(int parameterIndex){
        int classIndex = logisticRegression.getWeights().getClassIndex(parameterIndex);
        int[] labels = dataSet.getLabels();
        int featureIndex = logisticRegression.getWeights().getFeatureIndex(parameterIndex);
        double count = 0;
        //bias
        if (featureIndex == -1){
            for (int i=0;i<dataSet.getNumDataPoints();i++){
                if (labels[i]==classIndex){
                    count +=1;
                }
            }
        } else {
            Vector featureColumn = dataSet.getColumn(featureIndex);
            for (Vector.Element element: featureColumn.nonZeroes()){
                int dataPointIndex = element.index();
                double featureValue = element.get();
                int label = labels[dataPointIndex];
                if (label==classIndex){
                    count += featureValue;
                }
            }
        }
        return count;
    }

    private double calPredictedCount(int parameterIndex){
        int classIndex = logisticRegression.getWeights().getClassIndex(parameterIndex);
        int featureIndex = logisticRegression.getWeights().getFeatureIndex(parameterIndex);
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

    private void updateClassProbs(int dataPointIndex){
        double[] probs = logisticRegression.predictClassProbs(dataSet.getRow(dataPointIndex));
        System.arraycopy(probs, 0, this.classProbMatrix[dataPointIndex], 0, dataSet.getNumClasses());
    }

    private void updateClassProbMatrix(){
        IntStream.range(0,dataSet.getNumDataPoints()).parallel()
                .forEach(this::updateClassProbs);
    }

    private void updataDataGradient(int dataPointIndex){
        double[] classProbs = this.classProbMatrix[dataPointIndex];
        int label = dataSet.getLabels()[dataPointIndex];
        for (int k=0;k<dataSet.getNumClasses();k++){
            if (k==label){
                this.dataGradientMatrix[k][dataPointIndex] = classProbs[k] - 1;
            } else {
                this.dataGradientMatrix[k][dataPointIndex] = classProbs[k];
            }
        }
    }

    private void updateDataGradientMatrix(){
        IntStream.range(0,dataSet.getNumDataPoints()).parallel()
                .forEach(this::updataDataGradient);
    }

    public double[] getDataGradient(int k){
        return dataGradientMatrix[k];
    }

    public double[] getClassProbs(int dataPointIndex){
        return classProbMatrix[dataPointIndex];
    }

}
