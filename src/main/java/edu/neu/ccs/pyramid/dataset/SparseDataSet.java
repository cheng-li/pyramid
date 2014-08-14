package edu.neu.ccs.pyramid.dataset;

import org.apache.mahout.math.RandomAccessSparseVector;
import org.apache.mahout.math.SparseColumnMatrix;
import org.apache.mahout.math.SparseRowMatrix;
import org.apache.mahout.math.Vector;

import java.util.Arrays;

/**
 * Created by chengli on 8/4/14.
 */
public class SparseDataSet extends AbstractDataSet implements DataSet{
    protected RandomAccessSparseVector[] rowMatrix;
    protected RandomAccessSparseVector[] columnMatrix;

    public SparseDataSet(int numDataPoints, int numFeatures) {
        super(numDataPoints,numFeatures);
        this.rowMatrix = new RandomAccessSparseVector[numDataPoints];
        for (int i=0;i<numDataPoints;i++){
            rowMatrix[i] = new RandomAccessSparseVector(numFeatures);
        }
        this.columnMatrix = new RandomAccessSparseVector[numFeatures];
        for (int j=0;j<numFeatures;j++){
            columnMatrix[j] = new RandomAccessSparseVector(numDataPoints);
        }
    }


    @Override
    public FeatureColumn getFeatureColumn(int featureIndex) {
        return new FeatureColumn() {
            @Override
            public int getFeatureIndex() {
                return featureIndex;
            }

            @Override
            public Vector getVector() {
                return columnMatrix[featureIndex];
            }

            @Override
            public FeatureSetting getSetting() {
                return featureSettings[featureIndex];
            }
        };
    }

    @Override
    public FeatureRow getFeatureRow(int dataPointIndex) {
        return new FeatureRow() {
            @Override
            public int getDataPointIndex() {
                return dataPointIndex;
            }

            @Override
            public Vector getVector() {
                return rowMatrix[dataPointIndex];
            }

            @Override
            public DataSetting getSetting() {
                return dataSettings[dataPointIndex];
            }
        };
    }

    @Override
    public synchronized void setFeatureValue(int dataPointIndex, int featureIndex, double featureValue) {
        this.rowMatrix[dataPointIndex].set(featureIndex,featureValue);
        this.columnMatrix[featureIndex].set(dataPointIndex,featureValue);
    }

    @Override
    public boolean isDense() {
        return false;
    }

}
