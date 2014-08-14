package edu.neu.ccs.pyramid.dataset;

import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.Vector;

/**
 * Created by chengli on 8/7/14.
 */
public class DenseDataSet extends AbstractDataSet implements DataSet{
    protected DenseVector[] rowMatrix;
    protected DenseVector[] columnMatrix;

    public DenseDataSet(int numDataPoints, int numFeatures) {
        super(numDataPoints,numFeatures);
        this.rowMatrix = new DenseVector[numDataPoints];
        for (int i=0;i<numDataPoints;i++){
            rowMatrix[i] = new DenseVector(numFeatures);
        }
        this.columnMatrix = new DenseVector[numFeatures];
        for (int j=0;j<numFeatures;j++){
            columnMatrix[j] = new DenseVector(numDataPoints);
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
    public void setFeatureValue(int dataPointIndex, int featureIndex, double featureValue) {
        this.rowMatrix[dataPointIndex].set(featureIndex,featureValue);
        this.columnMatrix[featureIndex].set(dataPointIndex,featureValue);
    }

    @Override
    public boolean isDense() {
        return true;
    }
}
