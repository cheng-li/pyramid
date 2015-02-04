package edu.neu.ccs.pyramid.dataset;

/**
 * Created by chengli on 2/3/15.
 */
public class GradientMatrix {
    private int numDataPoints;
    private int numClasses;
    private double[][] dataClass;
    private double[][] classData;
    private Objective objective;

    public GradientMatrix(int numDataPoints, int numClasses, Objective objective) {
        this.numDataPoints = numDataPoints;
        this.numClasses = numClasses;
        this.dataClass = new double[numDataPoints][numClasses];
        this.classData = new double[numClasses][numDataPoints];
        this.objective = objective;
    }

    public void setGradient(int dataPointIndex, int classIndex, double gradient){
        this.dataClass[dataPointIndex][classIndex] = gradient;
        this.classData[classIndex][dataPointIndex] = gradient;
    }

    public double[] getGradientsForData(int dataPointIndex){
        return dataClass[dataPointIndex];
    }

    public double[] getGradientsForClass(int classIndex){
        return classData[classIndex];
    }

    public static enum Objective{
        MAXIMIZE, MINIMIZE
    }
}
