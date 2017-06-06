package edu.neu.ccs.pyramid.core.dataset;

/**
 * Created by chengli on 2/3/15.
 */
public class ProbabilityMatrix {
    // num data * num classes
    private float[][] m;

    public ProbabilityMatrix(int numDataPoints, int numClasses) {
        this.m = new float[numDataPoints][numClasses];
    }

    public void setProbability(int dataPointIndex, int classIndex, double probability){
        this.m[dataPointIndex][classIndex] = (float)probability;
    }

    public float[] getProbabilitiesForData(int dataPointIndex){
        return m[dataPointIndex];
    }


    public void increment(int dataPointIndex, int classIndex, double increment){
        this.m[dataPointIndex][classIndex] += increment;
    }


}

