package edu.neu.ccs.pyramid.multilabel_classification.sampling;

import edu.neu.ccs.pyramid.classification.Classifier;
import edu.neu.ccs.pyramid.classification.boosting.lktb.LKTreeBoost;
import edu.neu.ccs.pyramid.dataset.LabelTranslator;
import edu.neu.ccs.pyramid.dataset.MultiLabel;
import edu.neu.ccs.pyramid.dataset.MultiLabelClfDataSet;
import edu.neu.ccs.pyramid.feature.FeatureList;
import edu.neu.ccs.pyramid.multilabel_classification.MultiLabelClassifier;
import edu.neu.ccs.pyramid.util.MathUtil;
import org.apache.commons.math3.distribution.BinomialDistribution;
import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.RandomAccessSparseVector;
import org.apache.mahout.math.Vector;

import java.io.*;
import java.util.*;
import java.util.stream.Collectors;
import java.util.stream.DoubleStream;
import java.util.stream.IntStream;

/**
 * Created by Rainicy on 9/7/15.
 */
public class GibbsSampling implements MultiLabelClassifier {
    private static final long serialVersionUID = 6757804404524817727L;

    private List<Classifier.ProbabilityEstimator> classifiers;
    private int numClasses;

    // gibbs sampling K times.
    private int K;
    // count the majority starting from the last K;
    private int lastK;

    // feature list and labelTranslator, optional
    private FeatureList featureList;
    private LabelTranslator labelTranslator;


    /**
     * Constructor for only giving number of class,
     * gibbs sampling K times, and counting for the lastK times.
     * @param numClasses
     * @param K
     * @param lastK
     */
    public GibbsSampling(int numClasses, int K, int lastK) {
        this.numClasses = numClasses;
        this.K = K;
        this.lastK = lastK;
        this.classifiers = new ArrayList<>(numClasses);
    }

    public void addClassifier(Classifier.ProbabilityEstimator classifier, int index) {
        this.classifiers.add(index, classifier);
    }


//    public void savePrediction(MultiLabelClfDataSet dataSet, String outputFolder,
//                               String predFile, String probFile) throws IOException {
//
//        File preds = new File(outputFolder, predFile);
//        File probs = new File(outputFolder, probFile);
//        BufferedWriter bwPreds = new BufferedWriter(new FileWriter(preds));
//        BufferedWriter bwProbs = new BufferedWriter(new FileWriter(probs));
//
//        // go through all data points
//        MultiLabel[] labels = dataSet.getMultiLabels();
//        for (int k=0; k<dataSet.getNumDataPoints(); k++) {
//            System.out.println("#point : " + k + "/" + dataSet.getNumDataPoints());
//            System.out.println("true: " + labels[k].toString());
//            // append string with ","
//            String stringLabels = StringUtils.join(labels[k].getMatchedLabels(), ',');
//
//            bwPreds.write(stringLabels + '\t');
//            bwProbs.write(stringLabels + '\t');
//
//            Vector toVector = extendFeatures(dataSet.getRow(k));
//            int[] labelFeatures = new int[numClasses];
//
//            // starting gibbs sampling
//            for (int i=0; i<K; i++) {
//                // for each gibbs sampling iteration, go through all classifiers
//                int[] arrayPreds = new int[numClasses];
//                double[] arrayProbs = new double[numClasses];
//
//                for (int j=0; j<numClasses; j++) {
//                    // binary classifier.
//                    Classifier.ProbabilityEstimator classifier = classifiers.get(j);
//                    // the probability of predicting 1.
//                    double prob = classifier.predictClassProbs(toVector)[1];
//                    // update if the current label is on or off.
//                    int pred = flipCoin(prob);
//                    labelFeatures[j] = pred;
//                    // update the toVector for next classifier use.
//                    updateVector(toVector, labelFeatures, j);
//
//                    arrayPreds[j] = pred;
//                    arrayProbs[j] = prob;
//                }
////                System.out.println(StringUtils.join(arrayPreds, ',') + " ");
//                System.out.println(StringUtils.join(arrayProbs, ',') + " ");
////                System.in.read();
//                bwPreds.write(StringUtils.join(arrayPreds, ',') + " ");
//                bwProbs.write(StringUtils.join(arrayProbs, ',') + " ");
//            }
//            System.out.println();
//            System.in.read();
//
//            bwPreds.newLine();
//            bwProbs.newLine();
//        }
//
//        bwPreds.close();
//        bwProbs.close();
//    }

//    public MultiLabel[] predict(MultiLabelClfDataSet dataSet){
//        List<MultiLabel> results = new LinkedList<>();
//        for (int i=0; i<dataSet.getNumDataPoints(); i++) {
//            MultiLabel pred = predict(dataSet.getRow(i), dataSet.getMultiLabels()[i]);
//            results.add(pred);
//            System.out.println("predict: " + i + " - " + pred.toString());
////            try {
////                System.in.read();
////            } catch (IOException e) {
////                e.printStackTrace();
////            }
//        }
//        return results.toArray(new MultiLabel[results.size()]);
//    }

    public MultiLabel[] predict(MultiLabelClfDataSet dataSet){
        MultiLabel[] labels = dataSet.getMultiLabels();
//        List<MultiLabel> results = new ArrayList<>(dataSet.getNumDataPoints());
//        for (int i = 0; i < dataSet.getNumDataPoints(); i++) {
//            results.add(predict(dataSet.getRow(i), labels[i]));
//        }
        List<MultiLabel> results = IntStream.range(0, dataSet.getNumDataPoints()).parallel()
                .mapToObj(i -> predict(dataSet.getRow(i), labels[i]))
                .collect(Collectors.toList());
        return results.toArray(new MultiLabel[results.size()]);
    }

    public MultiLabel[] predict(MultiLabelClfDataSet dataSet, MultiLabel[] labels){
        List<MultiLabel> results = IntStream.range(0, dataSet.getNumDataPoints()).parallel()
                .mapToObj(i -> predict(dataSet.getRow(i), labels[i]))
                .collect(Collectors.toList());
        return results.toArray(new MultiLabel[results.size()]);
    }

    public MultiLabel[] tuningPredict(MultiLabelClfDataSet dataSet, MultiLabel[] labels){
        List<MultiLabel> results = IntStream.range(0, dataSet.getNumDataPoints()).parallel()
                .mapToObj(i -> tuningPredict(dataSet.getRow(i), labels[i]))
                .collect(Collectors.toList());
        return results.toArray(new MultiLabel[results.size()]);
    }


    /**
     * Predict with given label, and regard the gibbs sampling as a tuning
     * methods.
     * @param vector
     * @param label
     * @return
     */
    public MultiLabel tuningPredict(Vector vector, MultiLabel label) {
        // record the votes at the end of each gibbs sampling iteration.
        // get the count for each combination of labels(MultiLabel).
        Map<MultiLabel, Integer> votes = new LinkedHashMap<>();

        // append the (#labels-1) as features to the original vector
        Vector toVector = extendFeatures(vector);
        // initial current label features value with given label
        int[] consistLabelFeatures = new int[numClasses];
        for (int i : label.getMatchedLabels()) {
            if (i >= numClasses) {
                continue;
            }
            consistLabelFeatures[i] = 1;
        }
        // starting gibbs sampling
        for (int i=0; i<K; i++) {
            int[] labelFeatures = consistLabelFeatures.clone();
            // for each gibbs sampling iteration, go through all classifiers
            for (int j=0; j<numClasses; j++) {
                updateVector(toVector, labelFeatures, j);
                // binary classifier.
                Classifier.ProbabilityEstimator classifier = classifiers.get(j);
                // the probability of predicting 1.
                double prob = classifier.predictClassProbs(toVector)[1];
                // update if the current label is on or off.
                labelFeatures[j] = flipCoin(prob);
            }
            // record current iteration of prediction at the lastK iterations
            MultiLabel multiLabel = transMultiLabel(labelFeatures);
            if (!votes.containsKey(multiLabel)) {
                votes.put(multiLabel, 1);
            } else {
                votes.put(multiLabel, votes.get(multiLabel)+1);
            }
        }
        // predict the majority of the MultiLabel in votes.
        MultiLabel predLabel = new MultiLabel();
        int maxVoteCount = Integer.MIN_VALUE;
        for (HashMap.Entry<MultiLabel, Integer> vote : votes.entrySet()) {
            int voteCount = vote.getValue();
            if (maxVoteCount <= voteCount) {
                maxVoteCount = voteCount;
                predLabel = vote.getKey();
            }
        }
        return predLabel;
    }

    /**
     * Prediction by initializing sampling with the given label.
     * @param vector
     * @param label
     * @return
     */
    public MultiLabel predict(Vector vector, MultiLabel label) {
        // record the votes at the end of each gibbs sampling iteration.
        // get the count for each combination of labels(MultiLabel).
        Map<MultiLabel, Integer> votes = new LinkedHashMap<>();

        // append the (#labels-1) as features to the original vector
        Vector toVector = extendFeatures(vector);
        // initial current label features value with given label
        int[] labelFeatures = new int[numClasses];
        for (int i : label.getMatchedLabels()) {
            if (i >= numClasses) {
                continue;
            }
            labelFeatures[i] = 1;
        }

        // initialize the reorders
        int[] reorder = MathUtil.randomRange(0, numClasses);
        System.out.printf("random3 ...");
        // K is number of instances to pick also burn-in iteration number.
        Map<Integer, Double> S = new HashMap<>();
        MultiLabel[] B = new MultiLabel[K+lastK];


        Map<Integer, HashMap<MultiLabel, Double>> cache = new HashMap<>();
        for (int i=0; i<numClasses; i++) {
            cache.put(i, new HashMap<>());
        }

        // starting gibbs sampling
//        int l = 0;
        for (int i=0; i<(K+lastK); i++) {
//            System.out.println("current i: " + i + "/"+(K+lastK));
            // for each gibbs sampling iteration, go through all classifiers
            for (int k=0; k<numClasses; k++) {
                int j = reorder[k];
                updateVector(toVector, labelFeatures, j);
                // the probability of predicting 1.
                double prob = classifiers.get(j).predictClassProbs(toVector)[1];
                // update if the current label is on or off.
                labelFeatures[j] = flipCoin(prob);

                if (i > K) {
                    double s = 1;
                    MultiLabel multiLabel = transMultiLabel(labelFeatures);
                    if (cache.get(j).containsKey(multiLabel)) {
                        s = cache.get(j).get(multiLabel);
                    } else {
                        for (int kk=0; kk<numClasses; kk++) {
                            updateVector(toVector, labelFeatures, kk);
                            s *= classifiers.get(kk).predictClassProbs(toVector)[1];
                        }
                        cache.get(j).put(multiLabel, s);
                    }
                    S.put(i, s);
                    B[i] = transMultiLabel(labelFeatures);
//                    if (l < K) {
//                        S[l] = s;
//                        B[l] = transMultiLabel(labelFeatures);
//                        l++;
//                    } else {
//                        final int[] index = new int[1];
//
//                        IntStream.range(0, S.length ).parallel().
//                                reduce((a,b)->S[a]>S[b]? b: a).
//                                ifPresent(ix -> index[0] = ix);
//                        double v = S[index[0]];
//                        if (v < s) {
//                            S[index[0]] = s;
//                            B[index[0]] = transMultiLabel(labelFeatures);
//                        }
//                    }
                }
            }
            // record current iteration of prediction at the lastK iterations
//            if ((K-i) <= lastK) {
//                MultiLabel multiLabel = transMultiLabel(labelFeatures);
//                if (!votes.containsKey(multiLabel)) {
//                    votes.put(multiLabel, 1);
//                } else {
//                    votes.put(multiLabel, votes.get(multiLabel)+1);
//                }
//            }
        }

        Map<Integer, Double> sortedMap = new TreeMap(new ValueComparator(S));
        sortedMap.putAll(S);
        int firstK = 0;
        int[] selectedIndex = new int[K];
        for (Map.Entry<Integer, Double> entry : sortedMap.entrySet()) {
            if (firstK < K) {
                int index = entry.getKey();
//                double s = entry.getValue();
                selectedIndex[firstK] = index;
                firstK++;
//                System.out.println("index: " + index);
//                System.out.println("s : " + s);
            } else {
                break;
            }
        }

        for (int i=0; i<selectedIndex.length; i++) {
            MultiLabel multiLabel = B[selectedIndex[i]];
            if (!votes.containsKey(multiLabel)) {
                votes.put(multiLabel, 1);
            } else {
                votes.put(multiLabel, votes.get(multiLabel)+1);
            }
        }

        // predict the majority of the MultiLabel in votes.
        MultiLabel predLabel = new MultiLabel();
        int maxVoteCount = Integer.MIN_VALUE;
        for (HashMap.Entry<MultiLabel, Integer> vote : votes.entrySet()) {
            int voteCount = vote.getValue();
            if (maxVoteCount <= voteCount) {
                maxVoteCount = voteCount;
                predLabel = vote.getKey();
            }
        }

        return predLabel;
    }


    @Override
    public MultiLabel predict(Vector vector) {

        // record the votes at the end of each gibbs sampling iteration.
        // get the count for each combination of labels(MultiLabel).
        Map<MultiLabel, Integer> votes = new LinkedHashMap<>();

        // append the (#labels-1) as features to the original vector
        Vector toVector = extendFeatures(vector);
        // initial current label features value
        int[] labelFeatures = new int[numClasses];


        // initialize the reorders
        int[] reorder = MathUtil.randomRange(0, numClasses);
        System.out.printf("random3 ...");
        // K is number of instances to pick also burn-in iteration number.
        Map<Integer, Double> S = new HashMap<>();
        MultiLabel[] B = new MultiLabel[K+lastK];


        Map<Integer, HashMap<MultiLabel, Double>> cache = new HashMap<>();
        for (int i=0; i<numClasses; i++) {
            cache.put(i, new HashMap<>());
        }

        // starting gibbs sampling
//        int l = 0;
        for (int i=0; i<(K+lastK); i++) {
//            System.out.println("current i: " + i + "/"+(K+lastK));
            // for each gibbs sampling iteration, go through all classifiers
            for (int k=0; k<numClasses; k++) {
                int j = reorder[k];
                updateVector(toVector, labelFeatures, j);
                // the probability of predicting 1.
                double prob = classifiers.get(j).predictClassProbs(toVector)[1];
                // update if the current label is on or off.
                labelFeatures[j] = flipCoin(prob);

                if (i > K) {
                    double s = 1;
                    MultiLabel multiLabel = transMultiLabel(labelFeatures);
                    if (cache.get(j).containsKey(multiLabel)) {
                        s = cache.get(j).get(multiLabel);
                    } else {
                        for (int kk=0; kk<numClasses; kk++) {
                            updateVector(toVector, labelFeatures, kk);
                            s *= classifiers.get(kk).predictClassProbs(toVector)[1];
                        }
                        cache.get(j).put(multiLabel, s);
                    }
                    S.put(i, s);
                    B[i] = transMultiLabel(labelFeatures);
//                    if (l < K) {
//                        S[l] = s;
//                        B[l] = transMultiLabel(labelFeatures);
//                        l++;
//                    } else {
//                        final int[] index = new int[1];
//
//                        IntStream.range(0, S.length ).parallel().
//                                reduce((a,b)->S[a]>S[b]? b: a).
//                                ifPresent(ix -> index[0] = ix);
//                        double v = S[index[0]];
//                        if (v < s) {
//                            S[index[0]] = s;
//                            B[index[0]] = transMultiLabel(labelFeatures);
//                        }
//                    }
                }
            }
            // record current iteration of prediction at the lastK iterations
//            if ((K-i) <= lastK) {
//                MultiLabel multiLabel = transMultiLabel(labelFeatures);
//                if (!votes.containsKey(multiLabel)) {
//                    votes.put(multiLabel, 1);
//                } else {
//                    votes.put(multiLabel, votes.get(multiLabel)+1);
//                }
//            }
        }

        Map<Integer, Double> sortedMap = new TreeMap(new ValueComparator(S));
        sortedMap.putAll(S);
        int firstK = 0;
        int[] selectedIndex = new int[K];
        for (Map.Entry<Integer, Double> entry : sortedMap.entrySet()) {
            if (firstK < K) {
                int index = entry.getKey();
//                double s = entry.getValue();
                selectedIndex[firstK] = index;
                firstK++;
//                System.out.println("index: " + index);
//                System.out.println("s : " + s);
            } else {
                break;
            }
        }

        for (int i=0; i<selectedIndex.length; i++) {
            MultiLabel multiLabel = B[selectedIndex[i]];
            if (!votes.containsKey(multiLabel)) {
                votes.put(multiLabel, 1);
            } else {
                votes.put(multiLabel, votes.get(multiLabel)+1);
            }
        }

        // predict the majority of the MultiLabel in votes.
        MultiLabel predLabel = new MultiLabel();
        int maxVoteCount = Integer.MIN_VALUE;
        for (HashMap.Entry<MultiLabel, Integer> vote : votes.entrySet()) {
            int voteCount = vote.getValue();
            if (maxVoteCount <= voteCount) {
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
        int vectorIndex = toVector.size() - numClasses + 1;
        for (int i=0; i<labelFeatures.length; i++) {
            if (i == j) {
                continue;
            }
            toVector.set(vectorIndex, labelFeatures[i]);
            vectorIndex += 1;
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
        // return 1.
        BinomialDistribution bd = new BinomialDistribution(1, prob);
//        bd.sa
        return bd.sample();
//        if (Math.random() < prob) {
//            return 1;
//        }
//        return 0;
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
            toVector = new RandomAccessSparseVector(newLength);
        }

        // copy the original vector to the new vector
        for (Vector.Element element : vector.nonZeroes()) {
            toVector.set(element.index(), element.get());
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


    /**
     * Returns the number of classes.
     * @return
     */
    public int getNumClasses() {
        return this.numClasses;
    }

    /**
     * set the times of gibbs sampling.
     * @param K
     */
    public void setK(int K) {
        this.K = K;
    }

    /**
     * set the lastK times when starting to count
     * the majority.
     * @param lastK
     */
    public void setLastK(int lastK) {
        this.lastK = lastK;
    }


    public static GibbsSampling deserialize(String file) throws Exception {
        return deserialize(new File(file));
    }

    public static GibbsSampling deserialize(File file) throws Exception {
        try(
                FileInputStream fileInputStream = new FileInputStream(file);
                BufferedInputStream bufferedInputStream = new BufferedInputStream(fileInputStream);
                ObjectInputStream objectInputStream = new ObjectInputStream(bufferedInputStream);
        ){
            GibbsSampling sampling = (GibbsSampling) objectInputStream.readObject();
            return sampling;
        }
    }

    public class ValueComparator implements Comparator {

        Map map;

        public ValueComparator(Map map){
            this.map = map;
        }
        public int compare(Object keyA, Object keyB){

            Comparable valueA = (Comparable) map.get(keyA);
            Comparable valueB = (Comparable) map.get(keyB);

//            System.out.println(valueA +" - "+valueB);

            return valueB.compareTo(valueA);

        }
    }
}