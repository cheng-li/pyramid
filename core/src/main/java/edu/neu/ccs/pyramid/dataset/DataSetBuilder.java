package edu.neu.ccs.pyramid.dataset;

/**
 * Created by chengli on 11/28/14.
 */
public class DataSetBuilder {
    private int numDataPoints = -1;
    private int numFeatures = -1;
    private boolean dense = true;
    private boolean missingValue = false;
    private Density density = Density.DENSE;

    public static DataSetBuilder getBuilder(){
        return new DataSetBuilder();
    }

    public DataSetBuilder numDataPoints(int numDataPoints) {
        this.numDataPoints = numDataPoints;
        return this;
    }

    public DataSetBuilder numFeatures(int numFeatures) {
        this.numFeatures = numFeatures;
        return this;
    }

    public DataSetBuilder dense(boolean dense) {
        this.dense = dense;
        return this;
    }

    public DataSetBuilder missingValue(boolean missingValue) {
        this.missingValue = missingValue;
        return this;
    }

    public DataSetBuilder density(Density density) {
        this.density = density;
        return this;
    }

    public DataSet build(){
        if (!valid()){
            throw new IllegalArgumentException("Illegal arguments");
        }
        DataSet dataSet = null;
        switch (density){
            case DENSE:
                dataSet = new DenseDataSet(numDataPoints,numFeatures,missingValue);
                break;
            case SPARSE_RANDOM:
                dataSet = new SparseDataSet(numDataPoints,numFeatures,missingValue);
                break;
            case SPARSE_SEQUENTIAL:
                dataSet = new SequentialSparseDataSet(numDataPoints,numFeatures,missingValue);
                break;
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
