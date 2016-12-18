package edu.neu.ccs.pyramid.eval;

import edu.neu.ccs.pyramid.classification.Classifier;
import edu.neu.ccs.pyramid.classification.logistic_regression.LogisticRegression;
import edu.neu.ccs.pyramid.classification.logistic_regression.Weights;
import edu.neu.ccs.pyramid.dataset.DataSet;
import edu.neu.ccs.pyramid.dataset.MultiLabel;
import edu.neu.ccs.pyramid.dataset.MultiLabelClfDataSet;
import edu.neu.ccs.pyramid.multilabel_classification.MultiLabelClassifier;
import edu.neu.ccs.pyramid.multilabel_classification.cbm.CBM;
import edu.neu.ccs.pyramid.util.ArgSort;
import edu.neu.ccs.pyramid.util.Pair;
import org.apache.commons.math3.util.FastMath;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.apache.mahout.math.Vector;
import org.apache.xpath.operations.Bool;

import java.util.Arrays;
import java.util.List;
import java.util.stream.IntStream;
import java.util.HashMap;
import java.util.Map;

/**
 * Created by chengli on 11/11/14.
 */
public class KLDivergence {
    private static final Logger logger = LogManager.getLogger();

    public static double kl(double[] trueDistribution, double[] estimatedDistribution){
        double r = 0;
        for (int i=0;i<trueDistribution.length;i++){
            if (trueDistribution[i]==0){
                r += 0;
            } else if (estimatedDistribution[i]==0){
                r = Double.POSITIVE_INFINITY;
                break;
            } else {
                r += trueDistribution[i]* (Math.log(trueDistribution[i])-Math.log(estimatedDistribution[i]));
            }
        }
        if (Double.isInfinite(r)&&logger.isDebugEnabled()){
            logger.debug("true distribution = "+ Arrays.toString(trueDistribution));
            logger.debug("estimated distribution = "+ Arrays.toString(estimatedDistribution));
        }

        if (Double.isNaN(r)){
            throw new RuntimeException("KL divergence between "+Arrays.toString(trueDistribution)+" and "+Arrays.toString(estimatedDistribution)+" is NaN");
        }
        return r;
    }

    public static double klGivenPLogQ(double[] targetDistribution, double[] logEstimatedDistribution){
        double r = 0;
        for (int i=0;i<targetDistribution.length;i++){
            // if ==0, don't change sum
            if (targetDistribution[i]!=0){
                r += targetDistribution[i]* (Math.log(targetDistribution[i])-logEstimatedDistribution[i]);
            }
        }
        return r;
    }

    public static double kl(Classifier.ProbabilityEstimator estimator, Vector vector, double[] targetDistribution){
        double[] logEstimation = estimator.predictLogClassProbs(vector);
        return KLDivergence.klGivenPLogQ(targetDistribution,logEstimation);
    }

    public static double kl(Classifier.ProbabilityEstimator estimator, DataSet dataSet,
                     double[][] targetDistributions, double[] weights) {
        double sum = 0.0;
        for(int n=0; n<dataSet.getNumDataPoints(); n++) {
            sum += weights[n] * kl(estimator, dataSet.getRow(n), targetDistributions[n]);
        }
        return sum;
    }

    public static double kl(Classifier.ProbabilityEstimator estimator, DataSet dataSet,
                     double[][] targetDistributions) {
        double[] weights = new double[dataSet.getNumDataPoints()];
        Arrays.fill(weights,1.0);
        return kl(estimator,dataSet,targetDistributions,weights);
    }

    // total KL
    public static double kl(MultiLabelClassifier.AssignmentProbEstimator multiLabelClassifier, MultiLabelClfDataSet dataSet){
        return IntStream.range(0, dataSet.getNumDataPoints())
                .mapToDouble(i-> multiLabelClassifier.predictLogAssignmentProb(dataSet.getRow(i), dataSet.getMultiLabels()[i]))
                .sum()*(-1);
    }

    // empirical KL
    public static double kl_conditional(MultiLabelClassifier.AssignmentProbEstimator multiLabelClassifier, MultiLabelClfDataSet dataSet){
        Map<MultiLabel, Integer> q_z = new HashMap<MultiLabel, Integer>();
        Map<MultiLabel, HashMap<MultiLabel, Integer>> q_yz = new HashMap<MultiLabel, HashMap<MultiLabel, Integer>>();

        // get overall empirical distribution
        for (int i = 0; i < dataSet.getNumDataPoints(); ++i) {
            MultiLabel z = new MultiLabel(dataSet.getRow(i));
            MultiLabel y = dataSet.getMultiLabels()[i];
            if (q_z.containsKey(z)) {
                q_z.put(z, q_z.get(z) + 1);
            } else {
                q_z.put(z, 1);
            }

            if (!q_yz.containsKey(z)) {
                q_yz.put(z, new HashMap<MultiLabel, Integer>());
            }
            if (q_yz.get(z).containsKey(y)) {
                q_yz.get(z).put(y, q_yz.get(z).get(y) + 1);
            } else {
                q_yz.get(z).put(y, 1);
            }
        }

        // compute kl divergence
        double kl = 0.0;
        for (Map.Entry<MultiLabel, Integer> e1 : q_z.entrySet()) {
            double kl_y = 0.0;
            for (Map.Entry<MultiLabel, Integer> e2 : q_yz.get(e1.getKey()).entrySet()) {
                double empirical_prob_yz = (double)e2.getValue() / (double)e1.getValue();
                double log_estimated_prob_yz = multiLabelClassifier.predictLogAssignmentProb(e1.getKey().toVector(dataSet.getNumFeatures()), e2.getKey());
                kl_y += empirical_prob_yz * (Math.log(empirical_prob_yz) - log_estimated_prob_yz);
            }
            double empirical_prob_z = (double)e1.getValue() / (double)dataSet.getNumDataPoints();
            kl += empirical_prob_z * kl_y;
        }

        // Printing information if needed
        int occur_threshold = 10;
        double marginal_threshold = 0.01;
        for (Map.Entry<MultiLabel, Integer> e1 : q_z.entrySet()) {
            double[] marginals1 = new double[dataSet.getNumFeatures()];
            for (Map.Entry<MultiLabel, Integer> e2 : q_yz.get(e1.getKey()).entrySet()) {
                double estimated_prob_yz = multiLabelClassifier.predictAssignmentProb(e1.getKey().toVector(dataSet.getNumFeatures()), e2.getKey());
                double empirical_prob_yz = (double)e2.getValue() / (double)e1.getValue();
                if (e1.getValue() >= occur_threshold) {
                    System.out.println("#z:" + e1.getValue()+ ",z=" +
                            e1.getKey().toStringWithExtLabels(dataSet.getLabelTranslator()) + "->{" +
                            e2.getKey().toStringWithExtLabels(dataSet.getLabelTranslator()) + "},#y:" +
                            e2.getValue() + ",p_y|z_empirical:" + empirical_prob_yz +
                            ",p_y|z_estimated:" + estimated_prob_yz);
                }
                for (int i = 0; i < dataSet.getNumFeatures(); i++) {
                    if (e2.getKey().matchClass(i)) {
                        marginals1[i] += e2.getValue();
                    }
                }
            }
            if (e1.getValue() >= occur_threshold) {
                double estimated_prob_zz = multiLabelClassifier.predictAssignmentProb(e1.getKey().toVector(dataSet.getNumFeatures()), e1.getKey());
                System.out.println("p(y=z|z)=" + estimated_prob_zz);
                CBM cbm = (CBM) multiLabelClassifier;
//                List<MultiLabel> sampled = cbm.samples(e1.getKey().toVector(dataSet.getNumFeatures()), 10);
//                for (int i = 0; i < sampled.size(); ++i) {
//                    double prob = multiLabelClassifier.predictAssignmentProb(e1.getKey().toVector(dataSet.getNumFeatures()), sampled.get(i));
//                    System.out.println(sampled.get(i).toStringWithExtLabels(dataSet.getLabelTranslator()) + ":" + prob);
//                }
                System.out.println("p_y|z_estimated marginals are: ");
                double[] marginals = cbm.predictClassProbs(e1.getKey().toVector(dataSet.getNumFeatures()));
                int[] order = ArgSort.argSortDescending(marginals);
                for (int i = 0; i < order.length; ++i) {
                    if (marginals[order[i]] > marginal_threshold) {
                        System.out.println(dataSet.getLabelTranslator().toExtLabel(order[i]) + ":" + marginals[order[i]]);
                    }
                }

                System.out.println("p_y|z_empirical marginals are: ");
                for (int i = 0; i < dataSet.getNumFeatures(); i++) {
                    marginals1[i] /= (double)e1.getValue();
                }
                int[] order1 = ArgSort.argSortDescending(marginals1);
                for (int i = 0; i < order1.length; ++i) {
                    if (marginals1[order1[i]] > marginal_threshold) {
                        System.out.println(dataSet.getLabelTranslator().toExtLabel(order1[i]) + ":" + marginals1[order1[i]]);
                    }
                }
            }
        }

//        System.out.println("LRs for each label:");
//        CBM cbm = (CBM) multiLabelClassifier;
//        Classifier.ProbabilityEstimator[] estimators = cbm.getBinaryClassifiers()[0];
//        for (int i = 0; i < estimators.length; i++) {
//            System.out.println("LR:" + dataSet.getLabelTranslator().toExtLabel(i));
//            LogisticRegression lr = (LogisticRegression)estimators[i];
//            System.out.println(lr);
////            Vector weight_vec = lr.getWeights().getWeightsWithoutBiasForClass(1);
////            double[] weights = new double[weight_vec.size()];
////            for (int j = 0; j < weight_vec.size(); j++) {
////                weights[j] = weight_vec.get(j);
////            }
////
////            System.out.println("bias:" + lr.getWeights().getBiasForClass(1));
////            int[] order2 = ArgSort.argSortDescending(weights);
////            for (int j = 0; j < order2.length; ++j) {
////                System.out.println(dataSet.getLabelTranslator().toExtLabel(order2[j]) + ":" + weights[order2[j]]);
////            }
//        }
        System.out.println("---");

        return kl;
    }
}
