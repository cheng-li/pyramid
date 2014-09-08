package edu.neu.ccs.pyramid.dataset;


/**
 * Created by chengli on 8/4/14.
 */
public class SparseDataSet extends AbstractDataSet implements DataSet{
    protected SparseFeatureRow[] featureRows;
    protected SparseFeatureColumn[] featureColumns;

    public SparseDataSet(int numDataPoints, int numFeatures) {
        super(numDataPoints,numFeatures);
        this.featureRows = new SparseFeatureRow[numDataPoints];
        for (int i=0;i<numDataPoints;i++){
            this.featureRows[i] = new SparseFeatureRow(i,numFeatures);
        }
        this.featureColumns = new SparseFeatureColumn[numFeatures];
        for (int j=0;j<numFeatures;j++){
            this.featureColumns[j] = new SparseFeatureColumn(j,numDataPoints);
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

    /**
     * must be synchronized, otherwise may get ArrayIndexOutOfBoundsException
     * @param dataPointIndex
     * @param featureIndex
     * @param featureValue
     */
    @Override
    public synchronized void setFeatureValue(int dataPointIndex, int featureIndex, double featureValue) {
        this.featureRows[dataPointIndex].getVector().set(featureIndex,featureValue);
        this.featureColumns[featureIndex].getVector().set(dataPointIndex,featureValue);
    }


    @Override
    public boolean isDense() {
        return false;
    }

}
