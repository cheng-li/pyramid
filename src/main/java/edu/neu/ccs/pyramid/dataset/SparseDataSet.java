package edu.neu.ccs.pyramid.dataset;


import org.apache.mahout.math.RandomAccessSparseVector;
import org.apache.mahout.math.Vector;

/**
 * Created by chengli on 8/4/14.
 */
public class SparseDataSet extends AbstractDataSet implements DataSet{
    protected transient RandomAccessSparseVector[] featureRows;
    protected transient RandomAccessSparseVector[] featureColumns;

    public SparseDataSet(int numDataPoints, int numFeatures, boolean missingValue) {
        super(numDataPoints,numFeatures,missingValue);
        this.featureRows = new RandomAccessSparseVector[numDataPoints];
        for (int i=0;i<numDataPoints;i++){
            this.featureRows[i] = new RandomAccessSparseVector(numFeatures);
        }
        this.featureColumns = new RandomAccessSparseVector[numFeatures];
        for (int j=0;j<numFeatures;j++){
            this.featureColumns[j] = new RandomAccessSparseVector(numDataPoints);
        }
    }

    @Override
    public Density density() {
        return Density.SPARSE_RANDOM;
    }

    @Override
    public Vector getColumn(int featureIndex) {
        return this.featureColumns[featureIndex];
    }

    @Override
    public Vector getRow(int dataPointIndex) {
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
        if ((!this.hasMissingValue()) && Double.isNaN(featureValue)){
            throw new IllegalArgumentException("missing value is not allowed in this data set");
        }
        this.featureRows[dataPointIndex].set(featureIndex, featureValue);
        this.featureColumns[featureIndex].set(dataPointIndex, featureValue);
    }


    @Override
    public boolean isDense() {
        return false;
    }




}
