package edu.neu.ccs.pyramid.clustering.bmm;

import edu.neu.ccs.pyramid.dataset.DataSet;
import edu.neu.ccs.pyramid.optimization.*;
import org.apache.commons.math3.distribution.BinomialDistribution;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.Vector;

import java.util.Arrays;

/**
 * Created by chengli on 9/12/15.
 */
public class BMMTrainer {
    private static final Logger logger = LogManager.getLogger();
    DataSet dataSet;
    /**
     * gammas[i][k] = probability of data i in cluster k
     */
    double[][] gammas;
    private int numClusters;
    BMM bmm;
    private Terminator terminator;

    public BMMTrainer(DataSet dataSet,int numClusters) {
        this.numClusters = numClusters;
        this.dataSet = dataSet;
        this.gammas = new double[dataSet.getNumDataPoints()][numClusters];
        this.bmm = new BMM(numClusters,dataSet.getNumFeatures());
        this.terminator = new Terminator();
    }

    public BMM train(){
        while (true){
            iterate();
            if (terminator.shouldTerminate()){
                break;
            }
        }

        return this.bmm;
    }

    void iterate(){
        eStep();
        if (logger.isDebugEnabled()){
            logger.debug("E step is done ");
        }
        if (logger.isDebugEnabled()){
            logger.debug("gammas = "+ Arrays.deepToString(gammas));
        }
        mStep();
        if (logger.isDebugEnabled()){
            logger.debug("M step is done ");
        }
        if (logger.isDebugEnabled()){
            logger.debug("bmm = "+ bmm);
        }
    }

    private void eStep(){
        for (int i=0;i<dataSet.getNumDataPoints();i++){
            updateGamma(i);
        }
    }

    private void mStep(){
        for (int k=0;k<numClusters;k++){
            updateCluster(k);
        }
        double objective = objective();
        terminator.add(objective);
    }

    /**
     *
     * @param k cluster index
     */
    private void updateCluster(int k){
        double nk = 0;
        for (int i=0;i<dataSet.getNumDataPoints();i++){
            nk += gammas[i][k];
        }
        Vector average = new DenseVector(dataSet.getNumFeatures());
        for (int i=0;i<dataSet.getNumDataPoints();i++){
            average = average.plus(dataSet.getRow(i).times(gammas[i][k]));
        }
        average = average.divide(nk);
        if (logger.isDebugEnabled()){
            logger.debug("average vector = "+average);
        }
        for (int d=0;d<dataSet.getNumFeatures();d++){
            bmm.distributions[k][d] = new BinomialDistribution(1,average.get(d));
        }

        bmm.mixtureCoefficients[k] = nk/dataSet.getNumDataPoints();
    }


    private void updateGamma(int dataPointIndex){
        if (logger.isDebugEnabled()){
            logger.debug("before update gamma, gamma = "+Arrays.toString(gammas[dataPointIndex]));
        }
        for (int k=0;k<numClusters;k++){
            gammas[dataPointIndex][k] = bmm.mixtureCoefficients[k]*bmm.probability(dataSet.getRow(dataPointIndex),k);
        }
        double denominator = 0;
        for (int k=0;k<numClusters;k++){
            denominator += gammas[dataPointIndex][k];
        }
        for (int k=0;k<numClusters;k++){
            gammas[dataPointIndex][k] = gammas[dataPointIndex][k]/denominator;
        }
        if (logger.isDebugEnabled()){
            logger.debug("after update gamma, gamma = "+Arrays.toString(gammas[dataPointIndex]));
        }
    }


    /**
     * use Bishop Eq 9.51 as objective; Eq 9.55 is unstable when p=0;
     * @return
     */
    private double objective(){
        double res = 0;
        for (int i=0;i<dataSet.getNumDataPoints();i++){
            res += objective(i);
        }
        return res;
    }


    /**
     *
     * @param i data point
     * @return
     */
    private double objective(int i){
        double res = 0;
        for (int k=0;k<numClusters;k++){
            res += bmm.mixtureCoefficients[k]*bmm.probability(dataSet.getRow(i),k);
        }
        return Math.log(res);
    }



}
