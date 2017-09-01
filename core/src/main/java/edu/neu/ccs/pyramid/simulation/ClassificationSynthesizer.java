package edu.neu.ccs.pyramid.simulation;

import edu.neu.ccs.pyramid.dataset.ClfDataSet;
import edu.neu.ccs.pyramid.dataset.ClfDataSetBuilder;
import edu.neu.ccs.pyramid.dataset.RegDataSet;
import edu.neu.ccs.pyramid.dataset.RegDataSetBuilder;
import edu.neu.ccs.pyramid.util.Sampling;
import org.apache.commons.math3.distribution.NormalDistribution;

/**
 * Created by chengli on 5/30/15.
 */
public class ClassificationSynthesizer {
    private int numDataPoints;
    private double noiseMean;
    private double noiseSD;
    private int numFeatures;
    private NormalDistribution noise;

    public ClfDataSet multivarLine(){

        ClfDataSet dataSet = ClfDataSetBuilder.getBuilder()
                .numDataPoints(numDataPoints)
                .numFeatures(numFeatures)
                .numClasses(2)
                .dense(true)
                .missingValue(false)
                .build();
        for (int i=0;i<numDataPoints;i++){
            for (int j=0;j<numFeatures;j++){
                double featureValue = Sampling.doubleUniform(0, 1);
                dataSet.setFeatureValue(i,j,featureValue);
            }
            double sum = 0;
            for (int j=0;j<numFeatures;j++){
                sum += dataSet.getRow(i).get(j);
            }
            sum += noise.sample();
            if (sum>=numFeatures/2.0){
                dataSet.setLabel(i,1);
            } else {
                dataSet.setLabel(i,0);
            }
        }
        return dataSet;
    }

    public static Builder getBuilder(){
        return new Builder();
    }

    public static class Builder {
        private int numDataPoints=100;
        private double noiseMean=0;
        private double noiseSD=0.001;
        private int numFeatures=2;

        public Builder setNumDataPoints(int numDataPoints) {
            this.numDataPoints = numDataPoints;
            return this;
        }

        public Builder setNoiseMean(double noiseMean) {
            this.noiseMean = noiseMean;
            return this;
        }

        public Builder setNoiseSD(double noiseSD) {
            this.noiseSD = noiseSD;
            return this;
        }

        public Builder setNumFeatures(int numFeatures) {
            this.numFeatures = numFeatures;
            return this;
        }

        public ClassificationSynthesizer build() {
            ClassificationSynthesizer classificationSynthesizer = new ClassificationSynthesizer();
            classificationSynthesizer.numDataPoints = this.numDataPoints;
            classificationSynthesizer.noiseMean = this.noiseMean;
            classificationSynthesizer.noiseSD = this.noiseSD;
            classificationSynthesizer.noise = new NormalDistribution(noiseMean,noiseSD);
            classificationSynthesizer.numFeatures = this.numFeatures;
            return classificationSynthesizer;
        }
    }
}
