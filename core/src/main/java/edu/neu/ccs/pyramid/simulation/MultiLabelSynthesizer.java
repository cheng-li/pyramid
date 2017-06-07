package edu.neu.ccs.pyramid.simulation;

import edu.neu.ccs.pyramid.dataset.MLClfDataSetBuilder;
import edu.neu.ccs.pyramid.dataset.MultiLabel;
import edu.neu.ccs.pyramid.dataset.MultiLabelClfDataSet;
import edu.neu.ccs.pyramid.multilabel_classification.Enumerator;
import edu.neu.ccs.pyramid.multilabel_classification.crf.CMLCRF;
import edu.neu.ccs.pyramid.multilabel_classification.crf.SamplingPredictor;
import edu.neu.ccs.pyramid.multilabel_classification.crf.SubsetAccPredictor;
import edu.neu.ccs.pyramid.util.Sampling;
import org.apache.commons.math3.distribution.EnumeratedIntegerDistribution;
import org.apache.commons.math3.distribution.IntegerDistribution;
import org.apache.commons.math3.distribution.MultivariateNormalDistribution;
import org.apache.commons.math3.distribution.NormalDistribution;
import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.Vector;

import java.util.List;

/**
 * Created by chengli on 12/1/15.
 */
public class MultiLabelSynthesizer {

    /**
     * 60: 1,0
     * 40:0,1
     * @return
     */
    public static MultiLabelClfDataSet randomBinary(){
        MultiLabelClfDataSet dataSet = MLClfDataSetBuilder.getBuilder().numFeatures(1)
                .numClasses(2)
                .numDataPoints(100)
                .build();

        for (int i=0;i<60;i++){
            dataSet.addLabel(i,0);
        }

        for (int i=60;i<100;i++){
            dataSet.addLabel(i,1);
        }

        return dataSet;
    }


    /**
     * 30: 1,1
     * 40: 1,0
     * 30: 0,1
     * @return
     */
    public static MultiLabelClfDataSet randomTwoLabels(){
        MultiLabelClfDataSet dataSet = MLClfDataSetBuilder.getBuilder().numFeatures(1)
                .numClasses(2)
                .numDataPoints(100)
                .build();

        for (int i=0;i<30;i++){
            dataSet.addLabel(i,0);
            dataSet.addLabel(i,1);
        }
        for (int i=30;i<70;i++){
            dataSet.addLabel(i,0);
        }

        for (int i=70;i<100;i++){
            dataSet.addLabel(i,1);
        }

        return dataSet;
    }

    /**
     * 30: 1,0,0
     * 40: 0,1,0
     * 30: 0,0,1
     * @return
     */
    public static MultiLabelClfDataSet randomMultiClass(){
        MultiLabelClfDataSet dataSet = MLClfDataSetBuilder.getBuilder().numFeatures(1)
                .numClasses(3)
                .numDataPoints(100)
                .build();

        for (int i=0;i<30;i++){
            dataSet.addLabel(i,0);
        }
        for (int i=30;i<70;i++){
            dataSet.addLabel(i,1);
        }

        for (int i=70;i<100;i++){
            dataSet.addLabel(i,2);
        }

        return dataSet;
    }


    public static MultiLabelClfDataSet flipOne(int numData, int numFeature, int numClass){
        MultiLabelClfDataSet dataSet = MLClfDataSetBuilder.getBuilder().numFeatures(numFeature)
                .numClasses(numClass)
                .numDataPoints(numData)
                .build();

        // generate weights
        Vector[] weights = new Vector[numClass];
        for (int k=0;k<numClass;k++){
            Vector vector = new DenseVector(numFeature);
            for (int j=0;j<numFeature;j++){
                vector.set(j,Sampling.doubleUniform(-1,1));
            }
            weights[k] = vector;
        }

        // generate features
        for (int i=0;i<numData;i++){
            for (int j=0;j<numFeature;j++){
                dataSet.setFeatureValue(i,j,Sampling.doubleUniform(-1, 1));
            }
        }

        // assign labels
        for (int i=0;i<numData;i++){
            for (int k=0;k<numClass;k++){
                double dot = weights[k].dot(dataSet.getRow(i));
                if (dot>=0){
                    dataSet.addLabel(i,k);
                }
            }
        }


        // flip
        for (int i=0;i<numData;i++){
            int toChange = Sampling.intUniform(0,numClass-1);
            MultiLabel label = dataSet.getMultiLabels()[i];
            if (label.matchClass(toChange)){
                label.removeLabel(toChange);
            } else {
                label.addLabel(toChange);
            }

        }


        return dataSet;
    }


    public static MultiLabelClfDataSet flipTwo(int numData, int numFeature, int numClass){
        MultiLabelClfDataSet dataSet = MLClfDataSetBuilder.getBuilder().numFeatures(numFeature)
                .numClasses(numClass)
                .numDataPoints(numData)
                .build();

        // generate weights
        Vector[] weights = new Vector[numClass];
        for (int k=0;k<numClass;k++){
            Vector vector = new DenseVector(numFeature);
            for (int j=0;j<numFeature;j++){
                vector.set(j,Sampling.doubleUniform(-1,1));
            }
            weights[k] = vector;
        }

        // generate features
        for (int i=0;i<numData;i++){
            for (int j=0;j<numFeature;j++){
                dataSet.setFeatureValue(i,j,Sampling.doubleUniform(-1, 1));
            }
        }

        // assign labels
        for (int i=0;i<numData;i++){
            for (int k=0;k<numClass;k++){
                double dot = weights[k].dot(dataSet.getRow(i));
                if (dot>=0){
                    dataSet.addLabel(i,k);
                }
            }
        }


        // flip
        for (int i=0;i<numData;i++){
            int toChange = Sampling.intUniform(0,numClass-1);
            MultiLabel label = dataSet.getMultiLabels()[i];
            if (label.matchClass(toChange)){
                label.removeLabel(toChange);
            } else {
                label.addLabel(toChange);
            }
            if (toChange==0){
                int another = Sampling.intUniform(1,numClass-1);
                if (label.matchClass(another)){
                    label.removeLabel(another);
                } else {
                    label.addLabel(another);
                }
            }

        }


        return dataSet;
    }


    /**
     * y0: w=(0,1)
     * y1: w=(1,1)
     * y2: w=(1,0)
     * y3: w=(1,-1)
     * @param numData
     * @return
     */
    public static MultiLabelClfDataSet flipOneNonUniform(int numData){
        int numClass = 4;
        int numFeature = 2;

        MultiLabelClfDataSet dataSet = MLClfDataSetBuilder.getBuilder().numFeatures(numFeature)
                .numClasses(numClass)
                .numDataPoints(numData)
                .build();

        // generate weights
        Vector[] weights = new Vector[numClass];
        for (int k=0;k<numClass;k++){
            Vector vector = new DenseVector(numFeature);
            weights[k] = vector;
        }

        weights[0].set(0,0);
        weights[0].set(1,1);

        weights[1].set(0, 1);
        weights[1].set(1, 1);

        weights[2].set(0, 1);
        weights[2].set(1, 0);

        weights[3].set(0,1);
        weights[3].set(1,-1);


        // generate features
        for (int i=0;i<numData;i++){
            for (int j=0;j<numFeature;j++){
                dataSet.setFeatureValue(i,j,Sampling.doubleUniform(-1, 1));
            }
        }

        // assign labels
        for (int i=0;i<numData;i++){
            for (int k=0;k<numClass;k++){
                double dot = weights[k].dot(dataSet.getRow(i));
                if (dot>=0){
                    dataSet.addLabel(i,k);
                }
            }
        }

        int[] indices = {0,1,2,3};
        double[] probs = {0.4,0.2,0.2,0.2};
        IntegerDistribution distribution = new EnumeratedIntegerDistribution(indices,probs);

        // flip
        for (int i=0;i<numData;i++){
            int toChange = distribution.sample();
            MultiLabel label = dataSet.getMultiLabels()[i];
            if (label.matchClass(toChange)){
                label.removeLabel(toChange);
            } else {
                label.addLabel(toChange);
            }

        }


        return dataSet;
    }


    /**
     * C0, y0: w=(0,1)
     * C0, y1: w=(1,1)
     * C1, y0: w=(1,0)
     * C1, y1: w=(1,-1)
     * @return
     */
    public static MultiLabelClfDataSet sampleFromMix(){
        int numData = 10000;
        int numClass = 2;
        int numFeature = 2;
        int numClusters = 2;
        double[] proportions = {0.4,0.6};
        int[] indices = {0,1};

        MultiLabelClfDataSet dataSet = MLClfDataSetBuilder.getBuilder()
                .numFeatures(numFeature)
                .numClasses(numClass)
                .numDataPoints(numData)
                .build();

        // generate weights
        Vector[][] weights = new Vector[numClusters][numClass];
        for (int c=0;c<numClusters;c++){
            for (int l=0;l<numClass;l++){
                Vector vector = new DenseVector(numFeature);
                weights[c][l] = vector;
            }
        }


        weights[0][0].set(0, 0);
        weights[0][0].set(1, 1);

        weights[0][1].set(0, 1);
        weights[0][1].set(1, 1);


        weights[1][0].set(0, 1);
        weights[1][0].set(1, 0);

        weights[1][1].set(0, 1);
        weights[1][1].set(1,-1);

        // generate features
        for (int i=0;i<numData;i++){
            for (int j=0;j<numFeature;j++){
                dataSet.setFeatureValue(i,j,Sampling.doubleUniform(-1, 1));
            }
        }
        IntegerDistribution distribution = new EnumeratedIntegerDistribution(indices,proportions);
        // assign labels
        for (int i=0;i<numData;i++){
            int cluster = distribution.sample();
            System.out.println("cluster "+cluster);
            for (int l=0;l<numClass;l++){
                System.out.println("row = "+dataSet.getRow(i));
                System.out.println("weight = "+ weights[cluster][l]);
                double dot = weights[cluster][l].dot(dataSet.getRow(i));
                System.out.println("dot = "+dot);
                if (dot>=0){
                    dataSet.addLabel(i,l);
                }
            }
        }

        return dataSet;
    }

    /**
     * y0: w=(0,1)
     * y1: w=(1,1)
     * y2: w=(1,0)
     * y3: w=(1,-1)
     * @return
     */
    public static MultiLabelClfDataSet independentNoise(){
        int numData = 10000;
        int numClass = 4;
        int numFeature = 2;

        MultiLabelClfDataSet dataSet = MLClfDataSetBuilder.getBuilder().numFeatures(numFeature)
                .numClasses(numClass)
                .numDataPoints(numData)
                .build();

        // generate weights
        Vector[] weights = new Vector[numClass];
        for (int k=0;k<numClass;k++){
            Vector vector = new DenseVector(numFeature);
            weights[k] = vector;
        }

        weights[0].set(0,0);
        weights[0].set(1,1);

        weights[1].set(0, 1);
        weights[1].set(1, 1);

        weights[2].set(0, 1);
        weights[2].set(1, 0);

        weights[3].set(0,1);
        weights[3].set(1,-1);


        // generate features
        for (int i=0;i<numData;i++){
            for (int j=0;j<numFeature;j++){
                dataSet.setFeatureValue(i,j,Sampling.doubleUniform(-1, 1));
            }
        }
        NormalDistribution[] noises = new NormalDistribution[4];
        noises[0] = new NormalDistribution(0,0.1);
        noises[1] = new NormalDistribution(0,0.1);
        noises[2] = new NormalDistribution(0,0.1);
        noises[3] = new NormalDistribution(0,0.1);

        // assign labels
        int numFlipped = 0;
        for (int i=0;i<numData;i++){
            for (int k=0;k<numClass;k++){
                double dot = weights[k].dot(dataSet.getRow(i));
                double score = dot + noises[k].sample();
                if (score>=0){
                    dataSet.addLabel(i,k);
                }
                if (dot*score<0){
                    numFlipped += 1;
                }
            }
        }

        System.out.println("number of flipped = "+numFlipped);
        return dataSet;
    }


    /**
     * 2 labels, 3 features, multi-variate gaussian noise
     * y0: w=(0,1,0)
     * y1: w=(1,0,0)
     * y2: w=(0,0,1)
     * @return
     */
    public static MultiLabelClfDataSet gaussianNoise(int numData){
        int numClass = 3;
        int numFeature = 3;

        MultiLabelClfDataSet dataSet = MLClfDataSetBuilder.getBuilder().numFeatures(numFeature)
                .numClasses(numClass)
                .numDataPoints(numData)
                .build();

        // generate weights
        Vector[] weights = new Vector[numClass];
        for (int k=0;k<numClass;k++){
            Vector vector = new DenseVector(numFeature);
            weights[k] = vector;
        }


        weights[0].set(1,1);

        weights[1].set(0, 1);


        weights[2].set(2, 1);



        // generate features
        for (int i=0;i<numData;i++){
            for (int j=0;j<numFeature;j++){
                dataSet.setFeatureValue(i,j,Sampling.doubleUniform(-1, 1));
            }
        }

        double[] means = new double[numClass];
        double[][] covars = new double[numClass][numClass];
        covars[0][0]=0.5;
        covars[0][1]=0.02;         covars[1][0]=0.02;
        covars[0][2]=-0.03;         covars[2][0]=-0.03;

        covars[1][1]=0.2;
        covars[1][2]=-0.03;         covars[2][1]=-0.03;



        covars[2][2]=0.3;

        MultivariateNormalDistribution distribution = new MultivariateNormalDistribution(means,covars);

        // assign labels
        int numFlipped = 0;
        for (int i=0;i<numData;i++){
            double[] noises = distribution.sample();
            for (int k=0;k<numClass;k++){
                double dot = weights[k].dot(dataSet.getRow(i));
                double score = dot + noises[k];
                if (score>=0){
                    dataSet.addLabel(i,k);
                }
                if (dot*score<0){
                    numFlipped += 1;
                }
            }
        }

        System.out.println("number of flipped bits = "+numFlipped);
        return dataSet;
    }


    /**
     * y0: w=(0,1)
     * y1: w=(1,1)
     * y2: w=(1,0)
     * y3: w=(1,-1)
     * @return
     */
    public static MultiLabelClfDataSet independent(){
        int numData = 10000;
        int numClass = 4;
        int numFeature = 2;

        MultiLabelClfDataSet dataSet = MLClfDataSetBuilder.getBuilder().numFeatures(numFeature)
                .numClasses(numClass)
                .numDataPoints(numData)
                .build();

        // generate weights
        Vector[] weights = new Vector[numClass];
        for (int k=0;k<numClass;k++){
            Vector vector = new DenseVector(numFeature);
            weights[k] = vector;
        }

        weights[0].set(0,0);
        weights[0].set(1,1);

        weights[1].set(0, 1);
        weights[1].set(1, 1);

        weights[2].set(0, 1);
        weights[2].set(1, 0);

        weights[3].set(0,1);
        weights[3].set(1,-1);


        // generate features
        for (int i=0;i<numData;i++){
            for (int j=0;j<numFeature;j++){
                dataSet.setFeatureValue(i,j,Sampling.doubleUniform(-1, 1));
            }
        }


        // assign labels
        for (int i=0;i<numData;i++){
            for (int k=0;k<numClass;k++){
                double dot = weights[k].dot(dataSet.getRow(i));
                double score = dot;
                if (score>=0){
                    dataSet.addLabel(i,k);
                }

            }
        }

        return dataSet;
    }


    public static MultiLabelClfDataSet crfSample(){
        int numData = 10000;
        int numClass = 4;
        int numFeature = 2;

        MultiLabelClfDataSet dataSet = MLClfDataSetBuilder.getBuilder().numFeatures(numFeature)
                .numClasses(numClass)
                .numDataPoints(numData)
                .build();

        List<MultiLabel> support = Enumerator.enumerate(numClass);
        CMLCRF cmlcrf = new CMLCRF(numClass, numFeature, support);

        cmlcrf.getWeights().getWeightsWithoutBiasForClass(0).set(0,0);
        cmlcrf.getWeights().getWeightsWithoutBiasForClass(0).set(1,10);

        cmlcrf.getWeights().getWeightsWithoutBiasForClass(1).set(0,10);
        cmlcrf.getWeights().getWeightsWithoutBiasForClass(1).set(1,10);

        cmlcrf.getWeights().getWeightsWithoutBiasForClass(2).set(0,10);
        cmlcrf.getWeights().getWeightsWithoutBiasForClass(2).set(1,0);

        cmlcrf.getWeights().getWeightsWithoutBiasForClass(3).set(0,-10);
        cmlcrf.getWeights().getWeightsWithoutBiasForClass(3).set(1,-10);


        // generate features
        for (int i=0;i<numData;i++){
            for (int j=0;j<numFeature;j++){
                dataSet.setFeatureValue(i,j,Sampling.doubleUniform(-1, 1));
            }
        }

        SamplingPredictor samplingPredictor = new SamplingPredictor(cmlcrf);

        // assign labels
        for (int i=0;i<numData;i++){
            MultiLabel label = samplingPredictor.predict(dataSet.getRow(i));
            dataSet.setLabels(i, label);
        }

        return dataSet;
    }

    public static MultiLabelClfDataSet crfArgmax(){
        int numData = 1000;
        int numClass = 4;
        int numFeature = 10;

        MultiLabelClfDataSet dataSet = MLClfDataSetBuilder.getBuilder().numFeatures(numFeature)
                .numClasses(numClass)
                .numDataPoints(numData)
                .build();

        List<MultiLabel> support = Enumerator.enumerate(numClass);
        CMLCRF cmlcrf = new CMLCRF(numClass, numFeature, support);

        cmlcrf.getWeights().getWeightsWithoutBiasForClass(0).set(0,0);
        cmlcrf.getWeights().getWeightsWithoutBiasForClass(0).set(1,10);

        cmlcrf.getWeights().getWeightsWithoutBiasForClass(1).set(0,10);
        cmlcrf.getWeights().getWeightsWithoutBiasForClass(1).set(1,10);

        cmlcrf.getWeights().getWeightsWithoutBiasForClass(2).set(0,10);
        cmlcrf.getWeights().getWeightsWithoutBiasForClass(2).set(1,0);

        cmlcrf.getWeights().getWeightsWithoutBiasForClass(3).set(0,-10);
        cmlcrf.getWeights().getWeightsWithoutBiasForClass(3).set(1,-10);


        // generate features
        for (int i=0;i<numData;i++){
            for (int j=0;j<numFeature;j++){
                dataSet.setFeatureValue(i,j,Sampling.doubleUniform(-1, 1));
            }
        }

        SubsetAccPredictor predictor = new SubsetAccPredictor(cmlcrf);

        // assign labels
        for (int i=0;i<numData;i++){
            MultiLabel label = predictor.predict(dataSet.getRow(i));
            dataSet.setLabels(i, label);
        }

        return dataSet;
    }

    public static MultiLabelClfDataSet crfArgmaxHide(){
        int numData = 10000;
        int numClass = 4;
        int numFeature = 2;

        MultiLabelClfDataSet dataSet = MLClfDataSetBuilder.getBuilder().numFeatures(numFeature)
                .numClasses(numClass)
                .numDataPoints(numData)
                .build();

        List<MultiLabel> support = Enumerator.enumerate(numClass);
        CMLCRF cmlcrf = new CMLCRF(numClass, numFeature, support);

        cmlcrf.getWeights().getWeightsWithoutBiasForClass(0).set(0,0);
        cmlcrf.getWeights().getWeightsWithoutBiasForClass(0).set(1,10);

        cmlcrf.getWeights().getWeightsWithoutBiasForClass(1).set(0,10);
        cmlcrf.getWeights().getWeightsWithoutBiasForClass(1).set(1,10);

        cmlcrf.getWeights().getWeightsWithoutBiasForClass(2).set(0,10);
        cmlcrf.getWeights().getWeightsWithoutBiasForClass(2).set(1,0);

        cmlcrf.getWeights().getWeightsWithoutBiasForClass(3).set(0,-10);
        cmlcrf.getWeights().getWeightsWithoutBiasForClass(3).set(1,-10);


        // generate features
        for (int i=0;i<numData;i++){
            for (int j=0;j<numFeature;j++){
                dataSet.setFeatureValue(i,j,Sampling.doubleUniform(-1, 1));
            }
        }

        SubsetAccPredictor predictor = new SubsetAccPredictor(cmlcrf);

        // assign labels
        for (int i=0;i<numData;i++){
            MultiLabel label = predictor.predict(dataSet.getRow(i));
            dataSet.setLabels(i, label);
        }


        // hide one feature
        for (int i=0;i<numData;i++){
            dataSet.setFeatureValue(i,0,0);

        }


        return dataSet;
    }

    public static MultiLabelClfDataSet crfArgmaxDrop(){
        int numData = 1000;
        int numClass = 4;
        int numFeature = 10;

        MultiLabelClfDataSet dataSet = MLClfDataSetBuilder.getBuilder().numFeatures(numFeature)
                .numClasses(numClass)
                .numDataPoints(numData)
                .build();

        List<MultiLabel> support = Enumerator.enumerate(numClass);
        CMLCRF cmlcrf = new CMLCRF(numClass, numFeature, support);

        cmlcrf.getWeights().getWeightsWithoutBiasForClass(0).set(0,0);
        cmlcrf.getWeights().getWeightsWithoutBiasForClass(0).set(1,10);

        cmlcrf.getWeights().getWeightsWithoutBiasForClass(1).set(0,10);
        cmlcrf.getWeights().getWeightsWithoutBiasForClass(1).set(1,10);

        cmlcrf.getWeights().getWeightsWithoutBiasForClass(2).set(0,10);
        cmlcrf.getWeights().getWeightsWithoutBiasForClass(2).set(1,0);

        cmlcrf.getWeights().getWeightsWithoutBiasForClass(3).set(0,-10);
        cmlcrf.getWeights().getWeightsWithoutBiasForClass(3).set(1,-10);


        // generate features
        for (int i=0;i<numData;i++){
            for (int j=0;j<numFeature;j++){
                dataSet.setFeatureValue(i,j,Sampling.doubleUniform(-1, 1));
            }
        }

        SubsetAccPredictor predictor = new SubsetAccPredictor(cmlcrf);


        // drop labels
        double[] alphas = {1, 0.9, 0.8, 0.7};

        // assign labels
        for (int i=0;i<numData;i++){
//            System.out.println(dataSet.getRow(i));
            MultiLabel label = predictor.predict(dataSet.getRow(i)).copy();
//            System.out.println(label);

            for (int l=0;l<numClass;l++){
                if (Math.random()>alphas[l] && label.matchClass(l)){
//                    System.out.println("drop");
                    label.removeLabel(l);
                }
            }

            dataSet.setLabels(i, label);
        }




        return dataSet;
    }
}
