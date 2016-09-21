package edu.neu.ccs.pyramid.dataset;

/**
 * Created by chengli on 9/20/16.
 */
public class FloatGradientMatrix {
    private float[][] classData;
    private FloatGradientMatrix.Objective objective;

    public FloatGradientMatrix(int numDataPoints, int numClasses, FloatGradientMatrix.Objective objective) {

        this.classData = new float[numClasses][numDataPoints];
        this.objective = objective;
    }

    public void setGradient(int dataPointIndex, int classIndex, double gradient){
        this.classData[classIndex][dataPointIndex] = (float)gradient;
    }

    public float[] getGradientsForClass(int classIndex){
        return classData[classIndex];
    }

    public static enum Objective{
        MAXIMIZE, MINIMIZE
    }
}
