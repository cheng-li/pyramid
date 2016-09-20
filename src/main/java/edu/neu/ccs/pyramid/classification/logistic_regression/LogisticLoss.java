package edu.neu.ccs.pyramid.classification.logistic_regression;

import edu.neu.ccs.pyramid.dataset.ClfDataSet;
import edu.neu.ccs.pyramid.dataset.DataSet;
import edu.neu.ccs.pyramid.dataset.GradientMatrix;
import edu.neu.ccs.pyramid.dataset.ProbabilityMatrix;
import edu.neu.ccs.pyramid.optimization.Optimizable;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.Vector;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.stream.IntStream;

/**
 * Created by Rainicy on 10/24/15.
 */
public class LogisticLoss implements Optimizable.ByGradientValue {
    private static final Logger logger = LogManager.getLogger();
    private LogisticRegression logisticRegression;
    private DataSet dataSet;
    // instance weights
    private double[] weights;
    private double[][] targetDistributions;
    private Vector empiricalCounts;
    private Vector predictedCounts;
    private Vector gradient;
    private int numParameters;
    private int numClasses;

    // size = num classes * num data
    private double[][] probabilityMatrix;

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
    private boolean isParallel = false;
    private double priorGaussianVariance;



    public LogisticLoss(LogisticRegression logisticRegression,
                        DataSet dataSet, double[] weights, double[][] targetDistributions,
                        double priorGaussianVariance, boolean parallel) {
        this.logisticRegression = logisticRegression;
        this.targetDistributions = targetDistributions;
        this.isParallel = parallel;
        numParameters = logisticRegression.getWeights().totalSize();
        this.dataSet = dataSet;
        this.weights = weights;
        this.priorGaussianVariance = priorGaussianVariance;
        this.empiricalCounts = new DenseVector(numParameters);
        this.predictedCounts = new DenseVector(numParameters);
        this.numClasses = targetDistributions[0].length;
        this.probabilityMatrix = new double[numClasses][dataSet.getNumDataPoints()];
        this.gradientMatrix = new GradientMatrix(dataSet.getNumDataPoints(),numClasses, GradientMatrix.Objective.MAXIMIZE);
        this.updateEmpricalCounts();
        this.isValueCacheValid=false;
        this.isGradientCacheValid=false;
    }



    public LogisticLoss(LogisticRegression logisticRegression,
                        DataSet dataSet, double[][] targetDistributions,
                        double gaussianPriorVariance, boolean parallel) {
        this(logisticRegression,dataSet,defaultWeights(dataSet.getNumDataPoints()),targetDistributions,gaussianPriorVariance, parallel);
    }


    public LogisticLoss(LogisticRegression logisticRegression,
                        ClfDataSet dataSet,
                        double gaussianPriorVariance, boolean parallel){
        this(logisticRegression,dataSet,defaultTargetDistribution(dataSet),gaussianPriorVariance, parallel);
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

        double kl = logisticRegression.dataSetKLWeightedDivergence(dataSet, targetDistributions, weights);
        if (logger.isDebugEnabled()){
            logger.debug("kl divergence = "+kl);
        }
        this.value =  kl + penaltyValue();
        this.isValueCacheValid = true;
        return this.value;
    }



    private double penaltyValue(int classIndex){
        double square = 0;
        Vector weightVector = logisticRegression.getWeights().getWeightsWithoutBiasForClass(classIndex);
        square += weightVector.dot(weightVector);
        return square/(2*priorGaussianVariance);
    }

    // total penalty
    public double penaltyValue(){
        IntStream intStream;
        if (isParallel){
            intStream = IntStream.range(0, numClasses).parallel();
        } else {
            intStream = IntStream.range(0, numClasses);
        }
        return intStream.mapToDouble(this::penaltyValue).sum();
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
        this.gradient = this.predictedCounts.minus(empiricalCounts).plus(penaltyGradient());
    }

    private Vector penaltyGradient(){
        Vector weightsVector = this.logisticRegression.getWeights().getAllWeights();
        Vector penalty = new DenseVector(weightsVector.size());

        penalty = penalty.plus(weightsVector.divide(priorGaussianVariance));

        for (int j:logisticRegression.getWeights().getAllBiasPositions()){
            penalty.set(j,0);
        }
        return penalty;
    }

    //todo removed isParallel
    private void updateEmpricalCounts(){
        IntStream intStream;
        if (isParallel){
            intStream = IntStream.range(0, numParameters).parallel();
        } else {
            intStream = IntStream.range(0, numParameters);
        }
        intStream.forEach(i -> this.empiricalCounts.set(i, calEmpricalCount(i)));
    }

    private void updatePredictedCounts(){
        IntStream intStream;
        if (isParallel){
            intStream = IntStream.range(0,numParameters).parallel();
        } else {
            intStream = IntStream.range(0,numParameters);
        }

        intStream.forEach(i -> this.predictedCounts.set(i, calPredictedCount(i)));
    }

    private double calEmpricalCount(int parameterIndex){
        int classIndex = logisticRegression.getWeights().getClassIndex(parameterIndex);
        int featureIndex = logisticRegression.getWeights().getFeatureIndex(parameterIndex);
        double count = 0;
        //bias
        if (featureIndex == -1){
            for (int i=0;i<dataSet.getNumDataPoints();i++){
                count += targetDistributions[i][classIndex]* weights[i];
            }
        } else {
            Vector featureColumn = dataSet.getColumn(featureIndex);
            for (Vector.Element element: featureColumn.nonZeroes()){
                int dataPointIndex = element.index();
                double featureValue = element.get();
                count += featureValue*targetDistributions[dataPointIndex][classIndex]* weights[dataPointIndex];
            }
        }
        return count;
    }

    private double calPredictedCount(int parameterIndex){
        int classIndex = logisticRegression.getWeights().getClassIndex(parameterIndex);
        int featureIndex = logisticRegression.getWeights().getFeatureIndex(parameterIndex);
        double count = 0;
        double[] probs = this.probabilityMatrix[classIndex];
        //bias
        if (featureIndex == -1){
            for (int i=0;i<dataSet.getNumDataPoints();i++){
                count += probs[i]* weights[i];
            }
        } else {
            Vector featureColumn = dataSet.getColumn(featureIndex);
            for (Vector.Element element: featureColumn.nonZeroes()){
                int dataPointIndex = element.index();
                double featureValue = element.get();
                count += probs[dataPointIndex]*featureValue* weights[dataPointIndex];
            }
        }
        return count;
    }

    private void updateClassProbs(int dataPointIndex){
        double[] probs = logisticRegression.predictClassProbs(dataSet.getRow(dataPointIndex));
        for (int k=0;k<numClasses;k++){
            this.probabilityMatrix[k][dataPointIndex]=probs[k];
        }
    }

    private void updateClassProbMatrix(){
        IntStream intStream;
        if (isParallel){
            intStream = IntStream.range(0,dataSet.getNumDataPoints()).parallel();
        } else {
            intStream = IntStream.range(0,dataSet.getNumDataPoints());
        }
        intStream.forEach(this::updateClassProbs);
    }





    public GradientMatrix getGradientMatrix() {
        return gradientMatrix;
    }

    private static double[] defaultWeights(int numDataPoints){
        double[] weights = new double[numDataPoints];
        Arrays.fill(weights,1.0);
        return weights;
    }


    private static double[][] defaultTargetDistribution(ClfDataSet dataSet){
        double[][] targetDistributions = new double[dataSet.getNumDataPoints()][dataSet.getNumClasses()];
        int[] labels = dataSet.getLabels();
        for (int i=0;i<labels.length;i++){
            int label = labels[i];
            targetDistributions[i][label]=1;
        }
        return targetDistributions;
    }




}
