package edu.neu.ccs.pyramid.dataset;

/**
 * Created by chengli on 9/20/16.
 */
public class FloatScoreMatrix {
    private float[][] dataClass;

    public FloatScoreMatrix(int numDataPoints, int numClasses) {

        this.dataClass = new float[numDataPoints][numClasses];
    }

    public void setScore(int dataPointIndex, int classIndex, double score){
        this.dataClass[dataPointIndex][classIndex] = (float)score;
    }

    public float[] getScoresForData(int dataPointIndex){
        return dataClass[dataPointIndex];
    }


    public void increment(int dataPointIndex, int classIndex, double increment){
        this.dataClass[dataPointIndex][classIndex] += increment;
    }
}
