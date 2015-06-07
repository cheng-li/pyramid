package edu.neu.ccs.pyramid.classification.logistic_regression;

import edu.neu.ccs.pyramid.dataset.ClfDataSet;
import edu.neu.ccs.pyramid.dataset.GradientMatrix;
import edu.neu.ccs.pyramid.dataset.ProbabilityMatrix;
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
public class LogisticLoss implements Optimizable.ByGradientValue{
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

    public LogisticLoss(LogisticRegression logisticRegression,
                        ClfDataSet dataSet, double gaussianPriorVariance) {
        this.logisticRegression = logisticRegression;
        numParameters = logisticRegression.getWeights().totalSize();
        this.dataSet = dataSet;
        this.gaussianPriorVariance = gaussianPriorVariance;
        this.empiricalCounts = new DenseVector(numParameters);
        this.predictedCounts = new DenseVector(numParameters);
        this.probabilityMatrix = new ProbabilityMatrix(dataSet.getNumDataPoints(),dataSet.getNumClasses());
        this.gradientMatrix = new GradientMatrix(dataSet.getNumDataPoints(),dataSet.getNumClasses(), GradientMatrix.Objective.MAXIMIZE);
        this.updateEmpricalCounts();
        this.isValueCacheValid=false;
        this.isGradientCacheValid=false;
    }

    /**
     *
     * @param logisticRegression
     * @param dataSet
     * @param gaussianPriorVariance
     * @param empiricalCounts has nothing to do with parameters
     */
    private LogisticLoss(LogisticRegression logisticRegression,
                        ClfDataSet dataSet, double gaussianPriorVariance,
                        Vector empiricalCounts) {
        this.logisticRegression = logisticRegression;
        numParameters = logisticRegression.getWeights().totalSize();
        this.dataSet = dataSet;
        this.gaussianPriorVariance = gaussianPriorVariance;
        this.predictedCounts = new DenseVector(numParameters);
        this.probabilityMatrix = new ProbabilityMatrix(dataSet.getNumDataPoints(),dataSet.getNumClasses());
        this.gradientMatrix = new GradientMatrix(dataSet.getNumDataPoints(),dataSet.getNumClasses(), GradientMatrix.Objective.MAXIMIZE);
        this.empiricalCounts = empiricalCounts;
        this.isValueCacheValid=false;
        this.isGradientCacheValid=false;
    }


    public Vector getParameters(){
        return logisticRegression.getWeights().getAllWeights();
    }

    @Override
    public ByGradientValue newInstance(Vector parameters) {
        LogisticRegression newFunction = new LogisticRegression(this.logisticRegression.getNumClasses(),
                this.logisticRegression.getNumFeatures(),parameters);
        return new LogisticLoss(newFunction, this.dataSet, this.gaussianPriorVariance, this.empiricalCounts);

    }

    @Override
    public void setParameters(Vector parameters) {
        this.getParameters().assign(parameters);
        this.isValueCacheValid=false;
        this.isGradientCacheValid=false;

    }


    public double getValue() {
        if (isValueCacheValid){
            return this.value;
        }
        Vector parameters = getParameters();
        this.value =  -1*logisticRegression.dataSetLogLikelihood(dataSet) + parameters.dot(parameters)/(2*gaussianPriorVariance);
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
        double[] probs = this.probabilityMatrix.getProbabilitiesForClass(classIndex);
        //bias
        if (featureIndex == -1){
            for (int i=0;i<dataSet.getNumDataPoints();i++){
                count += probs[i];
            }
        } else {
            Vector featureColumn = dataSet.getColumn(featureIndex);
            for (Vector.Element element: featureColumn.nonZeroes()){
                int dataPointIndex = element.index();
                double featureValue = element.get();
                count += probs[dataPointIndex] * featureValue;
            }
        }
        return count;
    }

    private void updateClassProbs(int dataPointIndex){
        double[] probs = logisticRegression.predictClassProbs(dataSet.getRow(dataPointIndex));
        for (int k=0;k<dataSet.getNumClasses();k++){
            this.probabilityMatrix.setProbability(dataPointIndex,k,probs[k]);
        }
    }

    private void updateClassProbMatrix(){
        IntStream.range(0,dataSet.getNumDataPoints()).parallel()
                .forEach(this::updateClassProbs);
    }

    private void updataDataGradient(int dataPointIndex){
        double[] classProbs = this.probabilityMatrix.getProbabilitiesForData(dataPointIndex);
        int label = dataSet.getLabels()[dataPointIndex];
        for (int k=0;k<dataSet.getNumClasses();k++){
            if (k==label){
                this.gradientMatrix.setGradient(dataPointIndex,k,1 - classProbs[k]);
            } else {
                this.gradientMatrix.setGradient(dataPointIndex,k,0 - classProbs[k]);
            }
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
