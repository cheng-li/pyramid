package edu.neu.ccs.pyramid.dataset;


import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.Vector;

import java.io.IOException;

/**
 * Created by chengli on 8/7/14.
 */
class DenseDataSet extends AbstractDataSet implements DataSet{

    protected transient DenseVector[] featureRows;
    protected transient DenseVector[] featureColumns;


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


    private void writeObject(java.io.ObjectOutputStream out)
            throws IOException {
        out.defaultWriteObject();
        SerializableVector[] serFeatureRows = new SerializableVector[featureRows.length];
        for (int i=0;i<featureRows.length;i++){
            serFeatureRows[i] = new SerializableVector(featureRows[i]);
        }
        SerializableVector[] serFeatureColumns = new SerializableVector[featureColumns.length];
        for (int i=0;i<featureColumns.length;i++){
            serFeatureColumns[i] = new SerializableVector(featureColumns[i]);
        }
        out.writeObject(serFeatureRows);
        out.writeObject(serFeatureColumns);
    }


    private void readObject(java.io.ObjectInputStream in)
            throws IOException, ClassNotFoundException{
        in.defaultReadObject();
        SerializableVector[] serFeatureRows = (SerializableVector[])in.readObject();
        featureRows = new DenseVector[serFeatureRows.length];
        for (int i=0;i<featureRows.length;i++){
            featureRows[i] = (DenseVector) serFeatureRows[i].getVector();
        }

        SerializableVector[] serFeatureColumns = (SerializableVector[])in.readObject();
        featureColumns = new DenseVector[serFeatureColumns.length];
        for (int i=0;i<featureColumns.length;i++){
            featureColumns[i] = (DenseVector) serFeatureColumns[i].getVector();
        }
    }


}
