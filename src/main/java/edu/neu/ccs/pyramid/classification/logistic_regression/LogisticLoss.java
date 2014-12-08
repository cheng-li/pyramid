package edu.neu.ccs.pyramid.classification.logistic_regression;

import edu.neu.ccs.pyramid.dataset.ClfDataSet;
import edu.neu.ccs.pyramid.optimization.Optimizable;
import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.Vector;

import java.util.stream.IntStream;


/**
 * Created by chengli on 12/7/14.
 */
public class LogisticLoss implements Optimizable.ByGradient{
    private LogisticRegression logisticRegression;
    private ClfDataSet dataSet;
    private double gaussianPriorVariance;
    private Vector empiricalCounts;
    private Vector predictedCounts;
    private Vector gradient;
    private int numParameters;
    private double[][] classProbs;

    public LogisticLoss(LogisticRegression logisticRegression,
                        ClfDataSet dataSet, double gaussianPriorVariance) {
        this.logisticRegression = logisticRegression;
        numParameters = logisticRegression.getWeights().totalSize();
        this.dataSet = dataSet;
        this.gaussianPriorVariance = gaussianPriorVariance;
        this.empiricalCounts = new DenseVector(numParameters);
        this.predictedCounts = new DenseVector(numParameters);
        this.classProbs = new double[dataSet.getNumDataPoints()][dataSet.getNumClasses()];
        this.updateEmpricalCounts();
    }

    public Vector getParameters(){
        return logisticRegression.getWeights().getAllWeights();
    }

    public Vector getGradient(){
        updateClassProbs();
        updatePredictedCounts();
        updateGradient();
        return this.gradient;
    }

    /**
     * gradient descent
     */
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
                count += this.classProbs[i][classIndex];
            }
        } else {
            Vector featureColumn = dataSet.getColumn(featureIndex);
            for (Vector.Element element: featureColumn.nonZeroes()){
                int dataPointIndex = element.index();
                double featureValue = element.get();
                count += this.classProbs[dataPointIndex][classIndex] * featureValue;
            }
        }
        return count;
    }

    private void updateClassProbs(int dataPointIndex){
        double[] probs = logisticRegression.predictClassProbs(dataSet.getRow(dataPointIndex));
        System.arraycopy(probs, 0, this.classProbs[dataPointIndex], 0, dataSet.getNumClasses());
    }

    private void updateClassProbs(){
        IntStream.range(0,dataSet.getNumDataPoints()).parallel()
                .forEach(this::updateClassProbs);
    }

}
