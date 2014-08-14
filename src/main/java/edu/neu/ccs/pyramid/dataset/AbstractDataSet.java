package edu.neu.ccs.pyramid.dataset;


/**
 * Created by chengli on 8/7/14.
 */
public abstract class AbstractDataSet implements DataSet{
    protected int numDataPoints;
    protected int numFeatures;
    protected DataSetting[] dataSettings;
    protected FeatureSetting[] featureSettings;

    public AbstractDataSet(int numDataPoints, int numFeatures) {
        this.numDataPoints = numDataPoints;
        this.numFeatures = numFeatures;
        this.dataSettings = new DataSetting[numDataPoints];
        //todo initialize
        this.featureSettings = new FeatureSetting[numFeatures];
        for (int i=0;i<numFeatures;i++){
            featureSettings[i] = new FeatureSetting();
        }
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
