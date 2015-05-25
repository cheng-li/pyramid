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
    static double noiseMean = 0;
    static double noiseSD = 0.00001;
    private static NormalDistribution noise = new NormalDistribution(noiseMean,noiseSD);



    public static RegDataSet univarStep(){
        int numDataPoints = 100;

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


    public static RegDataSet univarSine(){
        int numDataPoints = 100;

        RegDataSet dataSet = RegDataSetBuilder.getBuilder()
                .numDataPoints(numDataPoints)
                .numFeatures(1)
                .dense(true)
                .missingValue(false)
                .build();
        for (int i=0;i<numDataPoints;i++){
            double featureValue = Sampling.doubleUniform(-Math.PI/2,Math.PI/2);
            double label;
            label = Math.sin(featureValue);
            label += noise.sample();
            dataSet.setFeatureValue(i,0,featureValue);
            dataSet.setLabel(i,label);
        }
        return dataSet;
    }

    public static RegDataSet univarLine(){
        int numDataPoints = 100;

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

    public static RegDataSet univarQuadratic(){
        int numDataPoints = 100;

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

    public static RegDataSet univarExp(){
        int numDataPoints = 100;

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

    public static RegDataSet univarNormal(){
        int numDataPoints = 100;

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



    public static RegDataSet univarBeta(){
        int numDataPoints = 100;

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

    public static RegDataSet univarPiecewiseLinear(){
        int numDataPoints = 100;



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
}
