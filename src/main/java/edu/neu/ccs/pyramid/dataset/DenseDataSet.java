package edu.neu.ccs.pyramid.dataset;


/**
 * Created by chengli on 8/7/14.
 */
class DenseDataSet extends AbstractDataSet implements DataSet{

    protected DenseFeatureRow[] featureRows;
    protected DenseFeatureColumn[] featureColumns;

    DenseDataSet(int numDataPoints, int numFeatures) {
        super(numDataPoints,numFeatures);
        this.featureRows = new DenseFeatureRow[numDataPoints];
        for (int i=0;i<numDataPoints;i++){
            this.featureRows[i] = new DenseFeatureRow(i,numFeatures);
        }
        this.featureColumns = new DenseFeatureColumn[numFeatures];
        for (int j=0;j<numFeatures;j++){
            this.featureColumns[j] = new DenseFeatureColumn(j,numDataPoints);
        }
    }


    @Override
    public FeatureColumn getFeatureColumn(int featureIndex) {
        return this.featureColumns[featureIndex];
    }

    @Override
    public FeatureRow getFeatureRow(int dataPointIndex) {
        return this.featureRows[dataPointIndex];
    }

    @Override
    public void setFeatureValue(int dataPointIndex, int featureIndex, double featureValue) {
        this.featureRows[dataPointIndex].getVector().set(featureIndex,featureValue);
        this.featureColumns[featureIndex].getVector().set(dataPointIndex,featureValue);
    }


    @Override
    public boolean isDense() {
        return true;
    }


}
