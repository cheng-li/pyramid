package edu.neu.ccs.pyramid.dataset;

/**
 * Created by chengli on 11/1/14.
 */
public class MLClfDataSetBuilder {
    private int numDataPoints = -1;
    private int numFeatures = -1;
    private boolean missingValue = false;
    private int numClasses = -1;
    private Density density = Density.DENSE;

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


    public MLClfDataSetBuilder density(Density density) {
        this.density = density;
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

        MultiLabelClfDataSet dataSet = null;
        switch (density){
            case DENSE:
                dataSet = new DenseMLClfDataSet(numDataPoints,numFeatures,missingValue,numClasses);
                break;
            case SPARSE_RANDOM:
                dataSet = new SparseMLClfDataSet(numDataPoints,numFeatures,missingValue,numClasses);
                break;
            case SPARSE_SEQUENTIAL:
                dataSet = new SequentialSparseMLClfDataSet(numDataPoints,numFeatures,missingValue,numClasses);
                break;
        }

        return dataSet;
    }

}
