package edu.neu.ccs.pyramid.dataset;

/**
 * Created by chengli on 11/1/14.
 */
public class RegDataSetBuilder {
    private int numDataPoints = -1;
    private int numFeatures = -1;
    private boolean dense = true;
    private boolean missingValue = false;

    public static RegDataSetBuilder getBuilder(){
        return new RegDataSetBuilder();
    }

    public RegDataSetBuilder numDataPoints(int numDataPoints) {
        this.numDataPoints = numDataPoints;
        return this;
    }

    public RegDataSetBuilder numFeatures(int numFeatures) {
        this.numFeatures = numFeatures;
        return this;
    }

    public RegDataSetBuilder dense(boolean dense) {
        this.dense = dense;
        return this;
    }

    public RegDataSetBuilder missingValue(boolean missingValue) {
        this.missingValue = missingValue;
        return this;
    }



    public RegDataSet build(){
        if (!valid()){
            throw new IllegalArgumentException("Illegal arguments");
        }
        RegDataSet dataSet;
        if (dense){
            dataSet = new DenseRegDataSet(numDataPoints,numFeatures,missingValue);
        } else {
            dataSet = new SparseRegDataSet(numDataPoints,numFeatures,missingValue);
        }
        return dataSet;
    }

    private boolean valid(){
        if (numDataPoints<=0){
            return false;
        }

        if (numFeatures<=0){
            return false;
        }


        return true;
    }
}
