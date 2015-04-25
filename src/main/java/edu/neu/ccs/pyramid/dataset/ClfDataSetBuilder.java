package edu.neu.ccs.pyramid.dataset;


/**
 * Created by chengli on 11/1/14.
 */
public class ClfDataSetBuilder {
    private int numDataPoints = -1;
    private int numFeatures = -1;
    private boolean dense = true;
    private boolean missingValue = false;
    private int numClasses = -1;

    public static ClfDataSetBuilder getBuilder(){
        return new ClfDataSetBuilder();
    }

    public ClfDataSetBuilder numDataPoints(int numDataPoints) {
        this.numDataPoints = numDataPoints;
        return this;
    }

    public ClfDataSetBuilder numFeatures(int numFeatures) {
        this.numFeatures = numFeatures;
        return this;
    }

    public ClfDataSetBuilder dense(boolean dense) {
        this.dense = dense;
        return this;
    }

    public ClfDataSetBuilder missingValue(boolean missingValue) {
        this.missingValue = missingValue;
        return this;
    }

    public ClfDataSetBuilder numClasses(int numClasses) {
        this.numClasses = numClasses;
        return this;
    }

    public ClfDataSet build(){
        if (!valid()){
            throw new IllegalArgumentException("Illegal arguments");
        }
        ClfDataSet dataSet;
        if (dense){
            dataSet = new DenseClfDataSet(numDataPoints,numFeatures,missingValue,numClasses);
        } else {
            dataSet = new SparseClfDataSet(numDataPoints,numFeatures,missingValue,numClasses);
        }
        return dataSet;
    }

    private boolean valid(){
        if (numDataPoints<=0){
            System.out.println("numDataPoints<=0");
            return false;
        }

        if (numFeatures<=0){
            System.out.println("numFeatures<=0");
            return false;
        }

        if (numClasses<=0){
            System.out.println("numClasses<=0");
            return false;
        }

        return true;
    }
}
