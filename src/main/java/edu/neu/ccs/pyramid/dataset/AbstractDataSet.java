package edu.neu.ccs.pyramid.dataset;


/**
 * Created by chengli on 8/7/14.
 */
abstract class AbstractDataSet implements DataSet{
    protected int numDataPoints;
    protected int numFeatures;
    protected DataSetSetting settings;


    AbstractDataSet(int numDataPoints, int numFeatures) {
        this.numDataPoints = numDataPoints;
        this.numFeatures = numFeatures;
        this.settings = new DataSetSetting();
    }

    @Override
    public DataSetSetting getSetting() {
        return this.settings;
    }

    @Override
    public void putSetting(DataSetSetting setting) {
        this.settings = setting;
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
    public abstract FeatureColumn getFeatureColumn(int featureIndex);

    @Override
    public abstract FeatureRow getFeatureRow(int dataPointIndex);

    @Override
    public abstract void setFeatureValue(int dataPointIndex, int featureIndex, double featureValue);


    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder();
        sb.append("number of data points = ").append(numDataPoints).append("\n");
        sb.append("number of features = ").append(numFeatures).append("\n");
        sb.append("data settings:").append("\n");
        for (int i=0;i<numDataPoints;i++){
            sb.append(i).append(":").append(getFeatureRow(i).getSetting()).append(", ");
        }
        sb.append("\n");
        sb.append("feature settings:").append("\n");
        for (int i=0;i<numFeatures;i++){
            sb.append(i).append(":").append(getFeatureColumn(i).getSetting()).append(", ");
        }
        sb.append("\n");
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
        sb.append("\n");

        return sb.toString();
    }

    public String getMetaInfo(){
        StringBuilder sb = new StringBuilder();
        sb.append("data set meta information:").append("\n");
        sb.append("number of data points = ").append(getNumDataPoints()).append("\n");
        sb.append("number of features = ").append(getNumFeatures()).append("\n");
        return sb.toString();
    }
}
