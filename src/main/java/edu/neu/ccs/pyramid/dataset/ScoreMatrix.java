package edu.neu.ccs.pyramid.dataset;

/**
 * Created by chengli on 2/3/15.
 */
public class ScoreMatrix {
    private int numDataPoints;
    private int numClasses;
    private double[][] dataClass;
    private double[][] classData;

    public ScoreMatrix(int numDataPoints, int numClasses) {
        this.numDataPoints = numDataPoints;
        this.numClasses = numClasses;
        this.dataClass = new double[numDataPoints][numClasses];
        this.classData = new double[numClasses][numDataPoints];
    }

    public void setScore(int dataPointIndex, int classIndex, double score){
        this.dataClass[dataPointIndex][classIndex] = score;
        this.classData[classIndex][dataPointIndex] = score;
    }

    public double[] getScoresForData(int dataPointIndex){
        return dataClass[dataPointIndex];
    }

    public double[] getScoresForClass(int classIndex){
        return classData[classIndex];
    }

    public void increment(int dataPointIndex, int classIndex, double increment){
        this.dataClass[dataPointIndex][classIndex] += increment;
        this.classData[classIndex][dataPointIndex] += increment;
    }
}
