package edu.neu.ccs.pyramid.classification.logistic_regression;

import edu.neu.ccs.pyramid.dataset.ClfDataSet;
import edu.neu.ccs.pyramid.regression.linear_regression.ElasticNetLinearRegTrainer;
import edu.neu.ccs.pyramid.regression.linear_regression.LinearRegression;
import org.apache.mahout.math.Vector;

import java.util.Arrays;
import java.util.stream.IntStream;

/**
 * Friedman, Jerome, Trevor Hastie, and Rob Tibshirani.
 * "Regularization paths for generalized linear models via coordinate descent."
 * Journal of statistical software 33.1 (2010): 1.
 * Created by chengli on 2/24/15.
 */
public class ElasticNetLogisticTrainer {
    private double regularization;
    private double l1Ratio;
    // relative threshold
    private double epsilon;

    public static Builder getBuilder(){
        return new Builder();
    }

    public void train(LogisticRegression logisticRegression, ClfDataSet dataSet){
        double lastLoss = loss(logisticRegression,dataSet);
        double threshold = lastLoss*epsilon;
        while(true){
            iterate(logisticRegression,dataSet);
            double loss = loss(logisticRegression,dataSet);
            if (Math.abs(lastLoss-loss)<threshold){
                break;
            }
            lastLoss = loss;
        }
    }

    public void iterate(LogisticRegression logisticRegression,ClfDataSet dataSet){
        for (int k=0;k<dataSet.getNumClasses();k++){
            optimizeOneClass(logisticRegression,dataSet,k);
        }
    }

    private void optimizeOneClass(LogisticRegression logisticRegression,ClfDataSet dataSet,
                                  int classIndex){
        //create weighted least square problem
        int numDataPoints = dataSet.getNumDataPoints();
        double[] labels = new double[numDataPoints];
        double[] instanceWeights = new double[numDataPoints];
        IntStream.range(0,numDataPoints).parallel().forEach(i ->
        {
            double prob = logisticRegression.predictClassProbs(dataSet.getRow(i))[classIndex];
            double classScore = logisticRegression.predictClassScore(dataSet.getRow(i),classIndex);
            double y = 0;
            if (dataSet.getLabels()[i]==classIndex){
                y = 1;
            }
            double frac = 0;
            // if prob = 0 or prob = 1, weight = 0; doesn't matter how we decide frac; leave it 0
            if (prob!=0&&prob!=1){
                frac = (y-prob)/(prob*(1-prob));
            }
            // frac is numerically unstable; if it is too big, the weighted least square solver will crash
            if (frac>1){
                frac=1;
            }

            if (frac<-1){
                frac=-1;
            }

            labels[i] = classScore + frac;
            instanceWeights[i] = (prob*(1-prob))/numDataPoints;
        });

        LinearRegression linearRegression = new LinearRegression(dataSet.getNumFeatures(),
                logisticRegression.getWeights().getWeightsForClass(classIndex));
        // use default epsilon
        ElasticNetLinearRegTrainer linearRegTrainer = ElasticNetLinearRegTrainer.getBuilder()
                .setRegularization(this.regularization)
                .setL1Ratio(this.l1Ratio).build();
        linearRegTrainer.train(linearRegression,dataSet,labels,instanceWeights);
    }

    private double loss(LogisticRegression logisticRegression, ClfDataSet dataSet){
        double negativeLogLikelihood = logisticRegression.dataSetLogLikelihood(dataSet) * -1;
        double penalty = penalty(logisticRegression);
        return negativeLogLikelihood/dataSet.getNumDataPoints() + penalty;
    }

    private double penalty(LogisticRegression logisticRegression){
        double penalty = 0;
        for (int k=0;k<logisticRegression.getNumClasses();k++){
            Vector vector = logisticRegression.getWeights().getWeightsWithoutBiasForClass(k);
            double normCombination = (1-l1Ratio)*0.5*Math.pow(vector.norm(2),2) +
                    l1Ratio*vector.norm(1);
            penalty += regularization * normCombination;
        }
        return penalty;
    }

    public static class Builder{
        // when p>>N, logistic regression with 0 regularization is ill-defined
        // use a small regularization
        private double regularization=0.00001;
        private double l1Ratio=0;
        private double epsilon=0.001;

        public Builder setRegularization(double regularization) {
            boolean legal = regularization>=0;
            if (!legal){
                throw new IllegalArgumentException("regularization>=0");
            }
            this.regularization = regularization;
            return this;
        }


        public Builder setL1Ratio(double l1Ratio) {
            boolean legal = (l1Ratio>=0)&&(l1Ratio<=1);
            if (!legal){
                throw new IllegalArgumentException("(l1Ratio>=0)&&(l1Ratio<=1)");
            }
            this.l1Ratio = l1Ratio;
            return this;
        }

        public Builder setEpsilon(double epsilon) {
            boolean legal = (epsilon>0)&&(epsilon<1);
            if (!legal){
                throw new IllegalArgumentException("(epsilon>0)&&(epsilon<1)");
            }
            this.epsilon = epsilon;
            return this;
        }

        public ElasticNetLogisticTrainer build(){
            ElasticNetLogisticTrainer trainer = new ElasticNetLogisticTrainer();
            trainer.regularization = this.regularization;
            trainer.l1Ratio = this.l1Ratio;
            trainer.epsilon = this.epsilon;
            return trainer;
        }
    }
}
