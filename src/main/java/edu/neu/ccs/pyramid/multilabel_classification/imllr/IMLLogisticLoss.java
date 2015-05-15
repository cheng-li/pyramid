package edu.neu.ccs.pyramid.multilabel_classification.imllr;

import edu.neu.ccs.pyramid.dataset.MultiLabel;
import edu.neu.ccs.pyramid.dataset.MultiLabelClfDataSet;

import edu.neu.ccs.pyramid.optimization.Optimizable;
import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.Vector;

import java.util.stream.IntStream;

/**
 * Created by chengli on 5/15/15.
 */
public class IMLLogisticLoss implements Optimizable.ByGradient, Optimizable.ByGradientValue {
    private IMLLogisticRegression logisticRegression;
    private MultiLabelClfDataSet dataSet;
    private double gaussianPriorVariance;
    private Vector empiricalCounts;
    private Vector predictedCounts;
    private Vector gradient;
    private int numParameters;
    /**
     * numDataPoints by numClasses;
     */
    private double[][] classProbMatrix;


    public IMLLogisticLoss(IMLLogisticRegression mlLogisticRegression,
                          MultiLabelClfDataSet dataSet, double gaussianPriorVariance) {
        this.logisticRegression = mlLogisticRegression;
        numParameters = mlLogisticRegression.getWeights().totalSize();
        this.dataSet = dataSet;
        this.gaussianPriorVariance = gaussianPriorVariance;
        this.empiricalCounts = new DenseVector(numParameters);
        this.predictedCounts = new DenseVector(numParameters);
        this.classProbMatrix = new double[dataSet.getNumDataPoints()][dataSet.getNumClasses()];
        this.updateEmpricalCounts();
        this.refresh();
    }

    public Vector getParameters(){
        return logisticRegression.getWeights().getAllWeights();
    }

    public void refresh(){
        updateClassProbMatrix();
        updatePredictedCounts();
        updateGradient();
    }

    public double getValue(){
        return getValue(logisticRegression.getWeights().getAllWeights());
    }

    @Override
    public double getValue(Vector parameters) {
        IMLLogisticRegression tmpFunction = new IMLLogisticRegression(this.logisticRegression.getNumClasses(),
                this.logisticRegression.getNumFeatures(),this.logisticRegression.getAssignments(),parameters);
        return -tmpFunction.dataSetLogLikelihood(dataSet) + parameters.dot(parameters)/(2*gaussianPriorVariance);
    }

    public Vector getGradient(){
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
        MultiLabel[] labels = dataSet.getMultiLabels();
        int featureIndex = logisticRegression.getWeights().getFeatureIndex(parameterIndex);
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


    public double[] getClassProbs(int dataPointIndex){
        return classProbMatrix[dataPointIndex];
    }



    private void updateClassProbMatrix(){
        IntStream.range(0,dataSet.getNumDataPoints()).parallel()
                .forEach(i -> classProbMatrix[i] = logisticRegression.predictClassProbs(dataSet.getRow(i)));
    }
}
