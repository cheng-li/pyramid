package edu.neu.ccs.pyramid.dataset;


import org.apache.mahout.math.Vector;

import java.util.List;

/**
 * Created by chengli on 8/7/14.
 */
abstract class AbstractDataSet implements DataSet{
    protected int numDataPoints;
    protected int numFeatures;
    protected boolean missingValue;
    protected IdTranslator idTranslator;
    protected List<Feature> features;


    AbstractDataSet(int numDataPoints, int numFeatures, boolean missingValue) {
        this.numDataPoints = numDataPoints;
        this.numFeatures = numFeatures;
        this.missingValue = missingValue;
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
    public abstract Vector getColumn(int featureIndex);

    @Override
    public abstract Vector getRow(int dataPointIndex);

    @Override
    public abstract void setFeatureValue(int dataPointIndex, int featureIndex, double featureValue);

    @Override
    public boolean hasMissingValue() {
        return missingValue;
    }

    protected void allowMissingValue(){
        this.missingValue=true;
    }

    @Override
    public IdTranslator getIdTranslator() {
        return idTranslator;
    }

    @Override
    public List<Feature> getFeatureList() {
        return features;
    }

    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder();
        sb.append("number of data points = ").append(numDataPoints).append("\n");
        sb.append("number of features = ").append(numFeatures).append("\n");
        sb.append("has missing value = ").append(missingValue).append("\n");
        sb.append("row matrix:").append("\n");
        for (int i=0;i<numDataPoints;i++){
            sb.append(i).append(":\t").append(getRow(i).asFormatString()).append("\n");
        }
        sb.append("=====================================").append("\n");
        sb.append("column matrix:").append("\n");
        for (int j=0;j<numFeatures;j++){
            sb.append(j).append(":\t").append(getColumn(j).asFormatString()).append("\n");
        }
        sb.append("\n");

        return sb.toString();
    }

    public String getMetaInfo(){
        StringBuilder sb = new StringBuilder();
        sb.append("data set meta information:").append("\n");
        sb.append("number of data points = ").append(getNumDataPoints()).append("\n");
        sb.append("number of features = ").append(getNumFeatures()).append("\n");
        sb.append("has missing value = ").append(missingValue).append("\n");
        return sb.toString();
    }

    @Override
    public void setIdTranslator(IdTranslator idTranslator) {
        this.idTranslator = idTranslator;
    }

    @Override
    public void setFeatureList(List<Feature> featureList) {
        this.features = featureList;
    }
}
