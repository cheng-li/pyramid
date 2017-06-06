package edu.neu.ccs.pyramid.core.dataset;

/**
 * Created by chengli on 2/3/15.
 */
public class GradientMatrix {
    private int numDataPoints;
    private int numClasses;
    private double[][] classData;
    private Objective objective;

    public GradientMatrix(int numDataPoints, int numClasses, Objective objective) {
        this.numDataPoints = numDataPoints;
        this.numClasses = numClasses;
        this.classData = new double[numClasses][numDataPoints];
        this.objective = objective;
    }

    public void setGradient(int dataPointIndex, int classIndex, double gradient){
        this.classData[classIndex][dataPointIndex] = gradient;
    }


    public double[] getGradientsForClass(int classIndex){
        return classData[classIndex];
    }

    public static enum Objective{
        MAXIMIZE, MINIMIZE
    }
}
