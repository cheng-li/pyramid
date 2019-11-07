package edu.neu.ccs.pyramid.multilabel_classification.cbm;

import edu.neu.ccs.pyramid.dataset.MultiLabel;
import edu.neu.ccs.pyramid.multilabel_classification.DynamicProgramming;
import edu.neu.ccs.pyramid.util.MathUtil;

import java.util.*;

/**
 * Created by Rainicy on 11/27/15.
 */
public class CBMPredictor {


    /**
     * number of clusters
     */
    int numClusters;

    /**
     * number of labels
     */
    private int numLabels;

    /**
     * p[c|x]
     * cache for cluster probability
     */
    private double[] pi;

    /**
     * logP[c|x]
     * cache for cluster log probability
     */
    private double[] logPi;

    /**
     * log P[y|x, c]
     * cache for binary logistic regression for labels within each clusters.
     * format: logProbs[numClusters][numLabels][2]
     */
    private double[][][] logProbs;

    /**
     * P[y|x, c]
     * cache for binary logistic regression for labels within each clusters.
     * format: logProbs[numClusters][numLabels][2]
     */
    private double[][][] probs;

    /**
     * number of samples for sampling prediction method
     */
    private int numSample;

    /**
     * prediction allows empty or not.
     */
    private boolean allowEmpty = false;
    


    public CBMPredictor(BMDistribution bmDistribution) {
        this.numClusters = bmDistribution.numComponents;
        this.numLabels = bmDistribution.numLabels;
        this.pi = new double[numClusters];
        this.logPi = bmDistribution.logProportions;
        this.probs = new double[numClusters][numLabels][2];
        this.logProbs = new double[numClusters][numLabels][2];

        for (int k=0; k<numClusters; k++) {
            this.pi[k] = Math.exp(this.logPi[k]);
            for (int l = 0; l < numLabels; l++) {
                logProbs[k][l] = bmDistribution.logClassProbs[k][l];
                for (int i=0; i<2; i++) {
                    probs[k][l][i] = Math.exp(logProbs[k][l][i]);
                }
            }
        }
    }



    public MultiLabel predictByDynamic2() {
        // initialization
        Map<Integer, DynamicProgramming> DPs = new HashMap<>();
        double[] maxClusterProb = new double[numClusters];
        for (int k=0; k<numClusters; k++) {
            DPs.put(k,new DynamicProgramming(probs[k], logProbs[k]));
            maxClusterProb[k] = DPs.get(k).nextHighestProb();
        }


        // speed up:
        // 1) for pi^k (D^k - q) >= 1 - pi^k
        double[] cond1 = new double[numClusters];
        // 2) save condition for sum_{r!=k} (pi^r * D^r)
        double[] sumPiD = new double[numClusters];
        for (int k=0; k<numClusters; k++) {
            cond1[k] = maxClusterProb[k] - 1.0/ pi[k] + 1;
            double sum = 0.0;
            for (int r=0; r<numClusters; r++) {
                if (r == k) {
                    continue;
                }
                sum += pi[r] * maxClusterProb[r];
            }
            sumPiD[k] = sum;
        }

        double maxLogProb = Double.NEGATIVE_INFINITY;
        MultiLabel bestMultiLabel = new MultiLabel();

        int iter = 0;
        int maxIter = 10;

        while (DPs.size() > 0) {
            List<Integer> removeList = new LinkedList<>();
            for (Map.Entry<Integer, DynamicProgramming> entry : DPs.entrySet()) {
                int k = entry.getKey();
                DynamicProgramming dp = entry.getValue();
                double prob = dp.nextHighestProb();

                MultiLabel multiLabel = dp.nextHighestVector();

                // whether consider empty prediction
                if ((multiLabel.getNumMatchedLabels()==0) && !allowEmpty) {
                    if (dp.getQueue().size() == 0) {
                        removeList.add(k);
                    }
                    continue;
                }

                double logProb = logProbYnGivenXnLogisticProb(multiLabel);

                if (logProb >= maxLogProb) {
                    bestMultiLabel = multiLabel;
                    maxLogProb = logProb;
//                    maxIter = iter;
                }

                // check if need to remove cluster k from the candidates
                if (checkStop(prob, cond1[k], maxLogProb, sumPiD[k], k) || dp.getQueue().size() == 0) {
                    removeList.add(k);
                }
            }
            for (int k : removeList) {
                DPs.remove(k);
            }

            iter++;
            if (iter>=maxIter){
                break;
            }
        }

//
//        // loop break because of maximum iterations
//        if (iter == maxIter) {
//            MultiLabel sampleLabel = predictBySampling();
//            Vector sampleVector = new DenseVector(numLabels);
//            for (int l : sampleLabel.getMatchedLabels()) {
//                sampleVector.set(l, 1.0);
//            }
//            double sampleLogProb = logProbYnGivenXnLogisticProb(sampleVector);
//            if (sampleLogProb > maxLogProb) {
//                return sampleLabel;
//            }
//        }


        return bestMultiLabel;
    }


    private double computeThreshold(double[] componentProbs){
        double sum = 0;
        for (int k=0;k<numClusters;k++){
            sum += pi[k]*componentProbs[k];
        }
        return sum;
    }

    public MultiLabel predictByDynamic() {
        // initialization
        DynamicProgramming[] DPs = new DynamicProgramming[numClusters];
        double[] minClusterProb = new double[numClusters];
        for (int k=0; k<numClusters; k++) {
            DPs[k]=new DynamicProgramming(probs[k], logProbs[k]);
            minClusterProb[k] = Double.POSITIVE_INFINITY;
        }


        double maxLogProb = Double.NEGATIVE_INFINITY;
        MultiLabel bestMultiLabel = new MultiLabel();

        int maxIter = 10;

        for (int iter=0;iter<maxIter;iter++) {
            for (int k=0;k<numClusters;k++) {
                DynamicProgramming dp = DPs[k];
                double prob = dp.nextHighestProb();
                MultiLabel multiLabel = dp.nextHighestVector();

                minClusterProb[k] = prob;

                double threshold = computeThreshold(minClusterProb);

                boolean isCandidateValid = true;
                if ((multiLabel.getNumMatchedLabels()==0) && !allowEmpty){
                    isCandidateValid=false;
                }

                if (isCandidateValid){
                    double logProb = logProbYnGivenXnLogisticProb(multiLabel);
                    if (logProb >= maxLogProb) {
                        bestMultiLabel = multiLabel;
                        maxLogProb = logProb;
                    }
                }

                //stop condition

                if (Math.exp(maxLogProb)>=threshold){
                    return bestMultiLabel;
                }
            }
        }

        return bestMultiLabel;
    }



    private boolean checkStop(double q, double c1, double maxLogProb, double sumPiDk, int k) {
        if (q <= c1) {
            return true;
        }
        if (q * pi[k] <= Math.exp(maxLogProb) / numClusters) {
            return true;
        }
        if (pi[k] * q + sumPiDk <= Math.exp(maxLogProb)) {
            return true;
        }

        return false;
    }

    //todo fix
//    /**
//     * predict by sampling.
//     * @return
//     */
//    public MultiLabel predictBySampling() {
//
//        double maxLogProb = Double.NEGATIVE_INFINITY;
//        Vector predVector = new DenseVector(numLabels);
//
//        int[] clusters = IntStream.range(0, numClusters).toArray();
//        EnumeratedIntegerDistribution enumeratedIntegerDistribution = new EnumeratedIntegerDistribution(clusters, logisticProb);
//
//        for (int s=0; s<numSample; s++) {
//            int k = enumeratedIntegerDistribution.sample();
//            Vector candidateY = new DenseVector(numLabels);
//
//            for (int l=0; l<numLabels; l++) {
//                BernoulliDistribution bernoulliDistribution = new BernoulliDistribution(probs[k][l][1]);
//                candidateY.set(l, bernoulliDistribution.sample());
//            }
//            // whether consider empty prediction
//            if ((candidateY.maxValue() == 0.0) && !allowEmpty) {
//                continue;
//            }
//
//            double logProb = logProbYnGivenXnLogisticProb(candidateY);
//
//            if (logProb >= maxLogProb) {
//                predVector = candidateY;
//                maxLogProb = logProb;
//            }
//        }
//
//        MultiLabel predLabel = new MultiLabel();
//        for (int l=0; l<numLabels; l++) {
//            if (predVector.get(l) == 1.0) {
//                predLabel.addLabel(l);
//            }
//        }
//        return predLabel;
//    }

    /**
     * Predict by hard assignment for just one cluster with maximum prob.
     * @param
     * @return MultiLabl
     */
    public MultiLabel predictByHardAssignment() {
        // find the max cluster
        int maxK = 0;
        double maxPi = logPi[0];
        for (int k = 1; k< logPi.length; k++) {
            if (maxPi < logPi[k]) {
                maxK = k;
                maxPi = logPi[k];
            }
        }

        MultiLabel predict = new MultiLabel();
        for (int l=0; l<numLabels; l++) {
            if (probs[maxK][l][1] >= 0.5) {
                predict.addLabel(l);
            }
        }

        return predict;
    }

    private double logProbYnGivenXnLogisticProb(MultiLabel Y) {
        double[] logPYnk = clusterConditionalLogProbArr(Y);
        double[] sumLog = new double[logPi.length];
        for (int k=0; k<numClusters; k++) {
            sumLog[k] = logPi[k] + logPYnk[k];
        }

        return MathUtil.logSumExp(sumLog);
    }


    /**
     * return the log[p(y_n | z_n=k, x_n; w_k)] by all k from 1 to K.
     * @param Y
     * @return
     */
    public double[] clusterConditionalLogProbArr(MultiLabel Y) {
        double[] probArr = new double[numClusters];

        for (int k=0; k<numClusters; k++) {
            double logProb = 0.0;
            for (int l=0; l<numLabels; l++) {
                if (Y.matchClass(l)) {
                    logProb += logProbs[k][l][1];
                } else {
                    logProb += logProbs[k][l][0];
                }
                // short circuit
                if (logProb==Double.NEGATIVE_INFINITY){
                    break;
                }
            }
            probArr[k] = logProb;
        }

        return probArr;
    }



    public void setNumSamples(int numSample) {
        this.numSample = numSample;
    }

    public void setAllowEmpty(boolean allowEmpty) {
        this.allowEmpty = allowEmpty;
    }
}