package edu.neu.ccs.pyramid.multilabel_classification.bmm_variant;

import edu.neu.ccs.pyramid.classification.Classifier;
import edu.neu.ccs.pyramid.dataset.MultiLabel;
import edu.neu.ccs.pyramid.util.BernoulliDistribution;
import edu.neu.ccs.pyramid.util.MathUtil;
import org.apache.commons.math3.distribution.EnumeratedIntegerDistribution;
import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.Vector;

import java.io.IOException;
import java.util.*;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

/**
 * Created by Rainicy on 11/27/15.
 */
public class BMMPredictor {


    /**
     * number of clusters
     */
    int numClusters;

    /**
     * number of labels
     */
    int numLabels;

    /**
     * p[c|x]
     * cache for cluster probability, format:logisticProb[numClusters]
     */
    double[] logisticProb;

    /**
     * logP[c|x]
     * cache for cluster log probability, format:logisticLgProb[numClusters]
     */
    double[] logisticLogProb;

    /**
     * log P[y|x, c]
     * cache for binary logistic regression for labels within each clusters.
     * format: logProbs[numClusters][numLabels][2]
     */
    double[][][] logProbs;

    /**
     * P[y|x, c]
     * cache for binary logistic regression for labels within each clusters.
     * format: logProbs[numClusters][numLabels][2]
     */
    double[][][] probs;

    /**
     * number of samples for sampling prediction method
     */
    int numSample;

    /**
     * prediction allows empty or not.
     */
    boolean allowEmpty = false;

    /**
     * default
     * @param vector
     * @param multiNomialClassifiers
     * @param binaryClassifiers
     * @param numClusters
     * @param numLabels
     */
    public BMMPredictor(Vector vector, Classifier.ProbabilityEstimator multiNomialClassifiers,
                        Classifier.ProbabilityEstimator[][] binaryClassifiers, int numClusters, int numLabels) {
        this.numClusters = numClusters;
        this.numLabels = numLabels;
        this.logisticProb = new double[numClusters];
        this.logisticLogProb = multiNomialClassifiers.predictLogClassProbs(vector);
        this.probs = new double[numClusters][numLabels][2];
        this.logProbs = new double[numClusters][numLabels][2];

        for (int k=0; k<numClusters; k++) {
            this.logisticProb[k] = Math.exp(this.logisticLogProb[k]);
            for (int l = 0; l < numLabels; l++) {
                logProbs[k][l] = binaryClassifiers[k][l].predictLogClassProbs(vector);
                for (int i=0; i<2; i++) {
                    probs[k][l][i] = Math.exp(logProbs[k][l][i]);
                }
            }
        }
    }

    public MultiLabel predictByGreedy() {
        Vector predVector = new DenseVector(numLabels);
        double prevLogProb;
        if (allowEmpty) {
            prevLogProb = logProbYnGivenXnLogisticProb(predVector);
        } else {
            prevLogProb = Double.NEGATIVE_INFINITY;
        }

        Set<Integer> labelSet = IntStream.range(0, numLabels).boxed().collect(Collectors.toSet());
        while (!labelSet.isEmpty()) {
            double curLogProb = Double.NEGATIVE_INFINITY;
            int curLebel=-1;
            // find the maximum one
            for (int k : labelSet) {
                Vector curVector = new DenseVector((DenseVector) predVector, false);
                curVector.set(k, 1.0);
                double logProb = logProbYnGivenXnLogisticProb(curVector);
                if (logProb > curLogProb) {
                    curLebel = k;
                    curLogProb = logProb;
                }
            }

            if (curLogProb > prevLogProb) {
                predVector.set(curLebel, 1.0);
                prevLogProb = curLogProb;
                labelSet.remove(curLebel);
            } else {
                break;
            }
        }

        MultiLabel predLabel = new MultiLabel();
        for (int l=0; l<numLabels; l++) {
            if (predVector.get(l) == 1.0) {
                predLabel.addLabel(l);
            }
        }
        return predLabel;
    }

    public MultiLabel predictByDynamic() {
        // initialization
        Map<Integer, DynamicProgramming> DPs = new HashMap<>();
        double[] maxClusterProb = new double[numClusters];
        for (int k=0; k<numClusters; k++) {
            DPs.put(k,new DynamicProgramming(probs[k], logProbs[k]));
            maxClusterProb[k] = DPs.get(k).highestProb();
        }


        // speed up:
        // 1) for pi^k (D^k - q) >= 1 - pi^k
        double[] cond1 = new double[numClusters];
        // 2) save condition for sum_{r!=k} (pi^r * D^r)
        double[] sumPiD = new double[numClusters];
        for (int k=0; k<numClusters; k++) {
            cond1[k] = maxClusterProb[k] - 1.0/logisticProb[k] + 1;
            double sum = 0.0;
            for (int r=0; r<numClusters; r++) {
                if (r == k) {
                    continue;
                }
                sum += logisticProb[r] * maxClusterProb[r];
            }
            sumPiD[k] = sum;
        }

        double maxLogProb = Double.NEGATIVE_INFINITY;
        Vector predVector = new DenseVector(numLabels);

        int iter = 0;
        int maxIter = 0;

        while (DPs.size() > 0) {
            List<Integer> removeList = new LinkedList<>();
            for (Map.Entry<Integer, DynamicProgramming> entry : DPs.entrySet()) {
                int k = entry.getKey();
                DynamicProgramming dp = entry.getValue();
                double prob = dp.highestProb();

                Vector candidateY = dp.nextHighest();

                // whether consider empty prediction
                if ((candidateY.maxValue() == 0.0) && !allowEmpty) {
                    if (dp.dp.size() == 0) {
                        removeList.add(k);
                    }
                    continue;
                }

                double logProb = logProbYnGivenXnLogisticProb(candidateY);

                if (logProb >= maxLogProb) {
                    predVector = candidateY;
                    maxLogProb = logProb;
                    maxIter = iter;
                }

                // check if need to remove cluster k from the candidates
                if (checkStop(prob, cond1[k], maxLogProb, sumPiD[k], k) || dp.dp.size() == 0) {
                    removeList.add(k);
                }
            }
            for (int k : removeList) {
                DPs.remove(k);
            }

            iter++;
        }
//        System.out.println("maxIter: " + maxIter);
        MultiLabel predLabel = new MultiLabel();
        for (int l=0; l<numLabels; l++) {
            if (predVector.get(l) == 1.0) {
                predLabel.addLabel(l);
            }
        }
        return predLabel;
    }

    private boolean checkStop(double q, double c1, double maxLogProb, double sumPiDk, int k) {
        if (q <= c1) {
            return true;
        }
        if (q * logisticProb[k] <= Math.exp(maxLogProb) / numClusters) {
            return true;
        }
        if (logisticProb[k] * q + sumPiDk <= Math.exp(maxLogProb)) {
            return true;
        }

        return false;
    }

    /**
     * predict by sampling.
     * @return
     */
    public MultiLabel predictBySampling() {

        double maxLogProb = Double.NEGATIVE_INFINITY;
        Vector predVector = new DenseVector(numLabels);

        int[] clusters = IntStream.range(0, numClusters).toArray();
        EnumeratedIntegerDistribution enumeratedIntegerDistribution = new EnumeratedIntegerDistribution(clusters, logisticProb);

        for (int s=0; s<numSample; s++) {
            int k = enumeratedIntegerDistribution.sample();
            Vector candidateY = new DenseVector(numLabels);

            for (int l=0; l<numLabels; l++) {
                BernoulliDistribution bernoulliDistribution = new BernoulliDistribution(probs[k][l][1]);
                candidateY.set(l, bernoulliDistribution.sample());
            }
            // whether consider empty prediction
            if ((candidateY.maxValue() == 0.0) && !allowEmpty) {
                continue;
            }

            double logProb = logProbYnGivenXnLogisticProb(candidateY);

            if (logProb >= maxLogProb) {
                predVector = candidateY;
                maxLogProb = logProb;
            }
        }

        MultiLabel predLabel = new MultiLabel();
        for (int l=0; l<numLabels; l++) {
            if (predVector.get(l) == 1.0) {
                predLabel.addLabel(l);
            }
        }
        return predLabel;
    }

    private double logProbYnGivenXnLogisticProb(Vector Y) {
        double[] logPYnk = clusterConditionalLogProbArr(Y);
        double[] sumLog = new double[logisticLogProb.length];
        for (int k=0; k<numClusters; k++) {
            sumLog[k] = logisticLogProb[k] + logPYnk[k];
        }

        return MathUtil.logSumExp(sumLog);
    }


    /**
     * return the log[p(y_n | z_n=k, x_n; w_k)] by all k from 1 to K.
     * @param Y
     * @return
     */
    public double[] clusterConditionalLogProbArr(Vector Y) {
        double[] probArr = new double[numClusters];

        for (int k=0; k<numClusters; k++) {
            double logProb = 0.0;
            for (int l=0; l<numLabels; l++) {
                if (Y.get(l) == 1.0) {
                    logProb += logProbs[k][l][1];
                } else {
                    logProb += logProbs[k][l][0];
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
