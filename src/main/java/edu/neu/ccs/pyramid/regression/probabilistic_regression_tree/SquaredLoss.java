package edu.neu.ccs.pyramid.regression.probabilistic_regression_tree;

import edu.neu.ccs.pyramid.dataset.DataSet;
import edu.neu.ccs.pyramid.optimization.Optimizable;
import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.Vector;

import java.util.Arrays;
import java.util.stream.IntStream;

/**
 * Created by chengli on 5/21/15.
 */
public class SquaredLoss implements Optimizable.ByGradientValue {

    private DataSet dataSet;
    private double[] labels;
    /**
     * format:
     * bias, weights, left output, right output
     */
    private Vector vector;


    public SquaredLoss(DataSet dataSet, double[] labels) {
        this.dataSet = dataSet;
        this.labels = labels;
        this.vector = new DenseVector(dataSet.getNumFeatures() + 3);
    }

    public SquaredLoss(DataSet dataSet, double[] labels, Vector vector) {
        this.dataSet = dataSet;
        this.labels = labels;
        this.vector = vector;
    }

    private Vector getWeightsWithoutBias(){
        return vector.viewPart(1,vector.size()-3);
    }

    private double getBias(){
        return vector.get(0);
    }

    private double getLeftValue(){
        return vector.get(vector.size()-2);
    }

    private double getRightValue(){
        return vector.get(vector.size()-1);
    }


    @Override
    public Vector getGradient() {
        Vector gradient = new DenseVector(vector.size());

        Sigmoid sigmoid = new Sigmoid(getWeightsWithoutBias(),getBias());

        double leftValue = getLeftValue();
        double rightValue = getRightValue();
        double[] hx = new double[dataSet.getNumDataPoints()];
        IntStream.range(0,dataSet.getNumDataPoints()).parallel().forEach(
                i-> hx[i] = sigmoid.leftProbability(dataSet.getRow(i))
        );

        double[] fx = new double[dataSet.getNumDataPoints()];
        IntStream.range(0,dataSet.getNumDataPoints()).parallel().forEach(
                i-> fx[i] = hx[i]*leftValue + (1-hx[i])*rightValue
        );

        Vector d = new DenseVector(dataSet.getNumDataPoints());
        IntStream.range(0,dataSet.getNumDataPoints()).parallel().forEach(
                i-> d.set(i,(fx[i]-labels[i])*hx[i]*(1-hx[i]))
        );

        //gradient for bias
        gradient.set(0,(leftValue - rightValue)*d.zSum());

        //gradients for all features
        //assuming we want to update all weights
        IntStream.range(0,dataSet.getNumFeatures()).parallel().forEach(
                j -> gradient.set(j+1,(leftValue - rightValue)*d.dot(dataSet.getColumn(j)))
        );

        //gradient for left value
        double sumOfH = Arrays.stream(hx).parallel().sum();
        gradient.set(gradient.size()-2,sumOfH);
        //gradient for right value
        gradient.set(gradient.size()-1,dataSet.getNumDataPoints()-sumOfH);
        return gradient;
    }

    private double getValue(){
        Sigmoid sigmoid = new Sigmoid(getWeightsWithoutBias(),getBias());
        double leftValue = getLeftValue();
        double rightValue = getRightValue();
        double[] hx = new double[dataSet.getNumDataPoints()];
        IntStream.range(0,dataSet.getNumDataPoints()).parallel().forEach(
                i-> hx[i] = sigmoid.leftProbability(dataSet.getRow(i))
        );

        double[] fx = new double[dataSet.getNumDataPoints()];
        IntStream.range(0,dataSet.getNumDataPoints()).parallel().forEach(
                i-> fx[i] = hx[i]*leftValue + (1-hx[i])*rightValue
        );

        double squaredLoss = IntStream.range(0,dataSet.getNumDataPoints()).parallel()
                .mapToDouble(i -> Math.pow(fx[i]-labels[i],2)).sum()*0.5;
        return squaredLoss;
    }

    @Override
    public double getValue(Vector parameters) {
        SquaredLoss loss = new SquaredLoss(this.dataSet,this.labels,parameters);
        return loss.getValue();
    }

    @Override
    public Vector getParameters() {
        return vector;
    }

    @Override
    public void refresh() {

    }
}
