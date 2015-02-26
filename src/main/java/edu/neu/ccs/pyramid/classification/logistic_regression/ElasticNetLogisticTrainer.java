package edu.neu.ccs.pyramid.classification.logistic_regression;

import edu.neu.ccs.pyramid.dataset.ClfDataSet;
import edu.neu.ccs.pyramid.regression.linear_regression.ElasticNetLinearRegTrainer;
import edu.neu.ccs.pyramid.regression.linear_regression.LinearRegression;
import org.apache.mahout.math.Vector;

import java.util.stream.IntStream;

/**
 * Created by chengli on 2/24/15.
 */
public class ElasticNetLogisticTrainer {
    private double regularization;
    private double l1Ratio;
    //todo separate spsilon?
    private double epsilon;

    public static Builder getBuilder(){
        return new Builder();
    }

    public void train(LogisticRegression logisticRegression, ClfDataSet dataSet){
        double lastLoss = loss(logisticRegression,dataSet);
        while(true){
            iterate(logisticRegression,dataSet);
            double loss = loss(logisticRegression,dataSet);
            if (Math.abs(lastLoss-loss)<epsilon){
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
            if (prob==1 && y==0){
                throw new RuntimeException("prob==1 && y==0");
            }
            if (prob==0 && y==1){
                throw new RuntimeException("prob==0 && y==1");
            }
            if (prob==1 && y==1){
                frac = 1;
            } else if (prob==0 && y==0){
                frac = -1;
            } else {
                frac = (y-prob)/(prob*(1-prob));
            }


            labels[i] = classScore + frac;
            instanceWeights[i] = (prob*(1-prob))/numDataPoints;
        });

        LinearRegression linearRegression = new LinearRegression(dataSet.getNumFeatures(),
                logisticRegression.getWeights().getWeightsForClass(classIndex));
        ElasticNetLinearRegTrainer linearRegTrainer = ElasticNetLinearRegTrainer.getBuilder()
                .setEpsilon(this.epsilon).setRegularization(this.regularization)
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
        private double regularization=0;
        private double l1Ratio=0;
        private double epsilon=0.001;

        public Builder setRegularization(double regularization) {
            this.regularization = regularization;
            return this;
        }


        public Builder setL1Ratio(double l1Ratio) {
            this.l1Ratio = l1Ratio;
            return this;
        }

        public Builder setEpsilon(double epsilon) {
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
