package edu.neu.ccs.pyramid.multilabel_classification.sampling;

import edu.neu.ccs.pyramid.classification.Classifier;
import edu.neu.ccs.pyramid.dataset.LabelTranslator;
import edu.neu.ccs.pyramid.dataset.MultiLabel;
import edu.neu.ccs.pyramid.feature.FeatureList;
import edu.neu.ccs.pyramid.multilabel_classification.MultiLabelClassifier;
import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.SequentialAccessSparseVector;
import org.apache.mahout.math.Vector;

import java.util.*;

/**
 * Created by Rainicy on 9/7/15.
 */
public class GibbsSampling implements MultiLabelClassifier {

    private List<Classifier.ProbabilityEstimator> classifiers;
    private int numClasses;

    // gibbs sampling k times.
    private int K;

    // feature list and labelTranslator, optional
    private FeatureList featureList;
    private LabelTranslator labelTranslator;


    public GibbsSampling(int numClasses, int K) {
        this.numClasses = numClasses;
        this.K = K;
        for (int i=0; i<numClasses; i++) {
            this.classifiers = new ArrayList<>(numClasses);
        }
    }

    public void addClassifier(Classifier.ProbabilityEstimator classifier) {
        this.classifiers.add(classifier);
    }

    /**
     * Returns the number of classes.
     * @return
     */
    public int getNumClasses() {
        return this.numClasses;
    }

    @Override
    public MultiLabel predict(Vector vector) {

        // record the votes at the end of each gibbs sampling iteration.
        // get the count for each combination of labels(MultiLabel).
        Map<MultiLabel, Integer> votes = new HashMap<>();

        // append the (#labels-1) as features to the original vector
        Vector toVector = extendFeatures(vector);
        // initial current label features value
        int[] labelFeatures = new int[numClasses];


        // starting gibbs sampling
        for (int i=0; i<K; i++) {
            // for each gibbs sampling iteration, go through all classifiers
            for (int j=0; j<numClasses; j++) {
                // binary classifier.
                Classifier.ProbabilityEstimator classifier = classifiers.get(j);
                // the probability of predicting 1.
                double prob = classifier.predictClassProbs(toVector)[1];
                // update if the current label is on or off.
                labelFeatures[j] = flipCoin(prob);
                // update the toVector for next classifier use.
                updateVector(toVector, labelFeatures, j);
            }

            // record current iteration of prediction at the last 50 itertaions
            if ((K-i) <= 50) {
                MultiLabel multiLabel = transMultiLabel(labelFeatures);
                if (!votes.containsKey(multiLabel)) {
                    votes.put(multiLabel, 1);
                } else {
                    votes.put(multiLabel, votes.get(multiLabel)+1);
                }
            }
        }

        // predict the majority of the MultiLabel in votes.
        MultiLabel predLabel = null;
        int maxVoteCount = Integer.MIN_VALUE;
        for (HashMap.Entry<MultiLabel, Integer> vote : votes.entrySet()) {
            int voteCount = vote.getValue();
            if (maxVoteCount < voteCount) {
                maxVoteCount = voteCount;
                predLabel = vote.getKey();
            }
        }

        return predLabel;
    }

    /**
     * By given the array of label features, if the label's value=1, then turn on
     * this label, otherwise off.
     * @param labelFeatures
     * @return
     */
    private MultiLabel transMultiLabel(int[] labelFeatures) {
        MultiLabel multiLabel = new MultiLabel();

        for (int i=0; i<labelFeatures.length; i++) {
            if (labelFeatures[i] == 1) {
                multiLabel.addLabel(i);
            }
        }
        return multiLabel;
    }

    /**
     * update the given toVector based on given label features.
     * @param toVector
     * @param labelFeatures
     * @param j
     */
    private void updateVector(Vector toVector, int[] labelFeatures, int j) {

        int vectorIndex = numClasses;
        for (int i=0; i<labelFeatures.length; i++) {
            if (i == j) {
                continue;
            }
            toVector.set(vectorIndex++, labelFeatures[i]);
        }
    }

    /**
     * given the probability of header of a coin, simulate the
     * result of flipping the coin.
     * For example, given prob = 0.6: if generated prob = 0.3, then
     * return 1; if generated prob = 0.9, then return 0.
     * @param prob
     * @return 1 if generated prob is smaller than given prob; otherwise 0.
     */
    private int flipCoin(double prob) {
        // if the random probability is smaller than given prob,
        // return true.
        if (Math.random() < prob) {
            return 1;
        }
        return 0;
    }

    /**
     * By given a vector and number of classes, append (#classes-1) new features
     * to the original vector. The default new features values are 0.
     * @param vector the original features vector.
     * @return the extended vector.
     */
    private Vector extendFeatures(Vector vector) {
        Vector toVector;
        int newLength = vector.size() + numClasses - 1;

        // initialize the new vector
        if (vector.isDense()) {
            toVector = new DenseVector(newLength);
        } else {
            toVector = new SequentialAccessSparseVector(newLength);
        }

        // copy the original vector to the new vector
        for (Vector.Element element : vector.nonZeroes()) {
            int index = element.index();
            double value = element.get();
            toVector.set(index, value);
        }
        // append the label value as 0 to the new vector.
        for (int i=vector.size(); i<newLength; i++) {
            toVector.set(i, 0);
        }


        return toVector;
    }


    @Override
    public FeatureList getFeatureList() {
        return featureList;
    }

    @Override
    public LabelTranslator getLabelTranslator() {
        return labelTranslator;
    }

    public void setFeatureList(FeatureList featureList) {
        this.featureList = featureList;
    }

    public void setLabelTranslator(LabelTranslator labelTranslator) {
        this.labelTranslator = labelTranslator;
    }
}
