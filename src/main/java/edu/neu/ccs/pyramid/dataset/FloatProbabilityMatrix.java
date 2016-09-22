package edu.neu.ccs.pyramid.dataset;

/**
 * Created by chengli on 9/20/16.
 */
public class FloatProbabilityMatrix {
    private float[][] dataClass;

    public FloatProbabilityMatrix(int numDataPoints, int numClasses) {
        this.dataClass = new float[numDataPoints][numClasses];
    }

    public void setProbability(int dataPointIndex, int classIndex, double probability){
        this.dataClass[dataPointIndex][classIndex] = (float)probability;
    }

    public float[] getProbabilitiesForData(int dataPointIndex){
        return dataClass[dataPointIndex];
    }


    public void increment(int dataPointIndex, int classIndex, double increment){
        this.dataClass[dataPointIndex][classIndex] += increment;
    }
}
