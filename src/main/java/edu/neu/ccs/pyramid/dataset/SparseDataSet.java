package edu.neu.ccs.pyramid.dataset;

import org.apache.mahout.math.RandomAccessSparseVector;
import org.apache.mahout.math.SparseColumnMatrix;
import org.apache.mahout.math.SparseRowMatrix;
import org.apache.mahout.math.Vector;

import java.util.Arrays;

/**
 * Created by chengli on 8/4/14.
 */
public class SparseDataSet implements DataSet{
    protected int numDataPoints;
    protected int numFeatures;
    protected RandomAccessSparseVector[] rowMatrix;
    protected RandomAccessSparseVector[] columnMatrix;
    protected DataSetting[] dataSettings;
    protected FeatureSetting[] featureSettings;

    public SparseDataSet(int numDataPoints, int numFeatures) {
        this.numDataPoints = numDataPoints;
        this.numFeatures = numFeatures;
        this.rowMatrix = new RandomAccessSparseVector[numDataPoints];
        for (int i=0;i<numDataPoints;i++){
            rowMatrix[i] = new RandomAccessSparseVector(numFeatures);
        }
        this.columnMatrix = new RandomAccessSparseVector[numFeatures];
        for (int j=0;j<numFeatures;j++){
            columnMatrix[j] = new RandomAccessSparseVector(numDataPoints);
        }
        this.dataSettings = new DataSetting[numDataPoints];
        this.featureSettings = new FeatureSetting[numFeatures];
    }

    @Override
    public int getNumDataPoints() {
        return numDataPoints;
    }

    @Override
    public int getNumFeatures() {
        return numFeatures;
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
    public void putDataSetting(int dataPointIndex, DataSetting setting) {
        this.dataSettings[dataPointIndex] = setting;
    }

    @Override
    public void putFeatureSetting(int featureIndex, FeatureSetting setting) {
        this.featureSettings[featureIndex] = setting;
    }

    @Override
    public DataSetting getDataSetting(int dataPointIndex) {
        return this.dataSettings[dataPointIndex];
    }

    @Override
    public FeatureSetting getFeatureSetting(int featureIndex) {
        return this.featureSettings[featureIndex];
    }

    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder();
        sb.append("number of data points = ").append(numDataPoints).append("\n");
        sb.append("number of features = ").append(numFeatures).append("\n");
        sb.append("=====================================").append("\n");
        sb.append("row matrix:").append("\n");
        for (int i=0;i<numDataPoints;i++){
            sb.append(i).append(":\t").append(getFeatureRow(i).getVector().asFormatString()).append("\n");
        }
        sb.append("=====================================").append("\n");
        sb.append("column matrix:").append("\n");
        for (int j=0;j<numFeatures;j++){
            sb.append(j).append(":\t").append(getFeatureColumn(j).getVector().asFormatString()).append("\n");
        }

        return sb.toString();
    }
}
