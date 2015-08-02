package edu.neu.ccs.pyramid.dataset;

/**
 * Created by chengli on 11/1/14.
 */
public class MLClfDataSetBuilder {
    private int numDataPoints = -1;
    private int numFeatures = -1;
    private boolean dense = true;
    private boolean missingValue = false;
    private int numClasses = -1;

    public static MLClfDataSetBuilder getBuilder(){
        return new MLClfDataSetBuilder();
    }

    public MLClfDataSetBuilder numDataPoints(int numDataPoints) {
        this.numDataPoints = numDataPoints;
        return this;
    }

    public MLClfDataSetBuilder numFeatures(int numFeatures) {
        this.numFeatures = numFeatures;
        return this;
    }

    public MLClfDataSetBuilder dense(boolean dense) {
        this.dense = dense;
        return this;
    }

    public MLClfDataSetBuilder missingValue(boolean missingValue) {
        this.missingValue = missingValue;
        return this;
    }

    public MLClfDataSetBuilder numClasses(int numClasses) {
        this.numClasses = numClasses;
        return this;
    }

    public MultiLabelClfDataSet build(){
        if (numDataPoints<=0){
            throw new RuntimeException("numDataPoints<=0");
        }

        if (numFeatures<=0){
            throw new RuntimeException("numFeatures<=0");
        }

        if (numClasses<=0){
            throw new RuntimeException("numClasses<=0");
        }

        MultiLabelClfDataSet dataSet;
        if (dense){
            dataSet = new DenseMLClfDataSet(numDataPoints,numFeatures,missingValue,numClasses);
        } else {
            dataSet = new SparseMLClfDataSet(numDataPoints,numFeatures,missingValue,numClasses);
        }
        return dataSet;
    }

}
