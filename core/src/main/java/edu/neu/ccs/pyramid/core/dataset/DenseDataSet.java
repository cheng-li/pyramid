package edu.neu.ccs.pyramid.core.dataset;


import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.Vector;

/**
 * Created by chengli on 8/7/14.
 */
class DenseDataSet extends AbstractDataSet implements DataSet{

    protected DenseVector[] featureRows;
    protected DenseVector[] featureColumns;


    DenseDataSet(int numDataPoints, int numFeatures, boolean missingValue) {
        super(numDataPoints,numFeatures, missingValue);
        this.featureRows = new DenseVector[numDataPoints];
        for (int i=0;i<numDataPoints;i++){
            this.featureRows[i] = new DenseVector(numFeatures);
        }
        this.featureColumns = new DenseVector[numFeatures];
        for (int j=0;j<numFeatures;j++){
            this.featureColumns[j] = new DenseVector(numDataPoints);
        }
    }


    @Override
    public Vector getColumn(int featureIndex) {
        return this.featureColumns[featureIndex];
    }

    @Override
    public Vector getRow(int dataPointIndex) {
        return this.featureRows[dataPointIndex];
    }

    @Override
    public void setFeatureValue(int dataPointIndex, int featureIndex, double featureValue) {
        if ((!this.hasMissingValue()) && Double.isNaN(featureValue)){
            throw new IllegalArgumentException("missing value is not allowed in this data set");
        }
        this.featureRows[dataPointIndex].set(featureIndex, featureValue);
        this.featureColumns[featureIndex].set(dataPointIndex, featureValue);
    }


    @Override
    public boolean isDense() {
        return true;
    }


}
