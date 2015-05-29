package edu.neu.ccs.pyramid.simulation;

import edu.neu.ccs.pyramid.dataset.RegDataSet;
import edu.neu.ccs.pyramid.dataset.RegDataSetBuilder;
import edu.neu.ccs.pyramid.util.Sampling;
import org.apache.commons.math3.distribution.BetaDistribution;
import org.apache.commons.math3.distribution.NormalDistribution;

/**
 * Created by chengli on 5/21/15.
 */
public class RegressionSynthesizer {
    private int numDataPoints;
    private double noiseMean;
    private double noiseSD;
    private NormalDistribution noise;

    public static Builder getBuilder(){
        return new Builder();
    }

    public  RegDataSet univarStep(){


        RegDataSet dataSet = RegDataSetBuilder.getBuilder()
                .numDataPoints(numDataPoints)
                .numFeatures(1)
                .dense(true)
                .missingValue(false)
                .build();
        for (int i=0;i<numDataPoints;i++){
            double featureValue = Sampling.doubleUniform(0,1);
            double label;
            if (featureValue>0.5){
                label = 0.7;
            } else {
                label = 0.2;
            }
            label += noise.sample();
            dataSet.setFeatureValue(i,0,featureValue);
            dataSet.setLabel(i,label);
        }
        return dataSet;
    }


    public  RegDataSet univarSine(){


        RegDataSet dataSet = RegDataSetBuilder.getBuilder()
                .numDataPoints(numDataPoints)
                .numFeatures(1)
                .dense(true)
                .missingValue(false)
                .build();
        for (int i=0;i<numDataPoints;i++){
            double featureValue = Sampling.doubleUniform(-Math.PI,Math.PI);
            double label;
            label = Math.sin(featureValue);
            label += noise.sample();
            dataSet.setFeatureValue(i,0,featureValue);
            dataSet.setLabel(i,label);
        }
        return dataSet;
    }

    public  RegDataSet univarLine(){


        RegDataSet dataSet = RegDataSetBuilder.getBuilder()
                .numDataPoints(numDataPoints)
                .numFeatures(1)
                .dense(true)
                .missingValue(false)
                .build();
        for (int i=0;i<numDataPoints;i++){
            double featureValue = Sampling.doubleUniform(0,1);
            double label;
            label = featureValue;
            label += noise.sample();
            dataSet.setFeatureValue(i,0,featureValue);
            dataSet.setLabel(i,label);
        }
        return dataSet;
    }

    public  RegDataSet univarQuadratic(){


        RegDataSet dataSet = RegDataSetBuilder.getBuilder()
                .numDataPoints(numDataPoints)
                .numFeatures(1)
                .dense(true)
                .missingValue(false)
                .build();
        for (int i=0;i<numDataPoints;i++){
            double featureValue = Sampling.doubleUniform(0,1);
            double label;
            label = Math.pow(featureValue,2);
            label += noise.sample();
            dataSet.setFeatureValue(i,0,featureValue);
            dataSet.setLabel(i,label);
        }
        return dataSet;
    }

    public  RegDataSet univarExp(){


        RegDataSet dataSet = RegDataSetBuilder.getBuilder()
                .numDataPoints(numDataPoints)
                .numFeatures(1)
                .dense(true)
                .missingValue(false)
                .build();
        for (int i=0;i<numDataPoints;i++){
            double featureValue = Sampling.doubleUniform(0,1);
            double label;
            label = Math.exp(featureValue);
            label += noise.sample();
            dataSet.setFeatureValue(i,0,featureValue);
            dataSet.setLabel(i,label);
        }
        return dataSet;
    }

    public  RegDataSet univarNormal(){


        NormalDistribution normalDistribution = new NormalDistribution(0,1);

        RegDataSet dataSet = RegDataSetBuilder.getBuilder()
                .numDataPoints(numDataPoints)
                .numFeatures(1)
                .dense(true)
                .missingValue(false)
                .build();
        for (int i=0;i<numDataPoints;i++){
            double featureValue = Sampling.doubleUniform(-1,1);
            double label;
            label = normalDistribution.density(featureValue);
            label += noise.sample();
            dataSet.setFeatureValue(i,0,featureValue);
            dataSet.setLabel(i,label);
        }
        return dataSet;
    }



    public  RegDataSet univarBeta(){


        BetaDistribution betaDistribution = new BetaDistribution(2,5);

        RegDataSet dataSet = RegDataSetBuilder.getBuilder()
                .numDataPoints(numDataPoints)
                .numFeatures(1)
                .dense(true)
                .missingValue(false)
                .build();
        for (int i=0;i<numDataPoints;i++){
            double featureValue = Sampling.doubleUniform(0,1);
            double label;
            label =  betaDistribution.density(featureValue);
            label += noise.sample();
            dataSet.setFeatureValue(i,0,featureValue);
            dataSet.setLabel(i,label);
        }
        return dataSet;
    }

    public  RegDataSet univarPiecewiseLinear(){




        RegDataSet dataSet = RegDataSetBuilder.getBuilder()
                .numDataPoints(numDataPoints)
                .numFeatures(1)
                .dense(true)
                .missingValue(false)
                .build();
        for (int i=0;i<numDataPoints;i++){
            double featureValue = Sampling.doubleUniform(0,1);
            double label;
            if (featureValue<=0.5){
                label = - featureValue + 0.5;
            } else {
                label = featureValue;
            }
            label += noise.sample();
            dataSet.setFeatureValue(i,0,featureValue);
            dataSet.setLabel(i,label);
        }
        return dataSet;
    }




    public  RegDataSet multivarLine(){

        int numFeatures = 2;

        RegDataSet dataSet = RegDataSetBuilder.getBuilder()
                .numDataPoints(numDataPoints)
                .numFeatures(2)
                .dense(true)
                .missingValue(false)
                .build();
        for (int i=0;i<numDataPoints;i++){
            for (int j=0;j<numFeatures;j++){
                double featureValue = Sampling.doubleUniform(0,1);
                dataSet.setFeatureValue(i,j,featureValue);
            }
            double label = 0;
            for (int j=0;j<numFeatures;j++){
                label += dataSet.getRow(i).get(j);
            }
            label += noise.sample();
            dataSet.setLabel(i,label);
        }
        return dataSet;
    }

    public static class Builder {
        private int numDataPoints=100;
        private double noiseMean=0;
        private double noiseSD=0.001;

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

        public RegressionSynthesizer build() {
            RegressionSynthesizer regressionSynthesizer = new RegressionSynthesizer();
            regressionSynthesizer.numDataPoints = this.numDataPoints;
            regressionSynthesizer.noiseMean = this.noiseMean;
            regressionSynthesizer.noiseSD = this.noiseSD;
            regressionSynthesizer.noise = new NormalDistribution(noiseMean,noiseSD);
            return regressionSynthesizer;
        }
    }
}
