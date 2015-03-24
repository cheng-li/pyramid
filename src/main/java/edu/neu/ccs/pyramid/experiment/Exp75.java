package edu.neu.ccs.pyramid.experiment;

import edu.neu.ccs.pyramid.classification.PriorProbClassifier;
import edu.neu.ccs.pyramid.classification.logistic_regression.LogisticRegression;
import edu.neu.ccs.pyramid.classification.logistic_regression.LogisticRegressionInspector;
import edu.neu.ccs.pyramid.configuration.Config;
import edu.neu.ccs.pyramid.dataset.ClfDataSet;
import edu.neu.ccs.pyramid.dataset.DataSetType;
import edu.neu.ccs.pyramid.dataset.DataSetUtil;
import edu.neu.ccs.pyramid.dataset.TRECFormat;
import edu.neu.ccs.pyramid.feature.Feature;
import edu.neu.ccs.pyramid.feature.FeatureUtility;
import edu.neu.ccs.pyramid.feature.Ngram;
import edu.neu.ccs.pyramid.util.SetUtil;
import org.apache.mahout.math.Vector;

import java.io.File;
import java.util.*;
import java.util.stream.Collectors;

/**
 * Created by chengli on 3/23/15.
 */
public class Exp75 {
    public static void main(String[] args) throws Exception{
        if (args.length !=1){
            throw new IllegalArgumentException("please specify the config file");
        }

        Config config = new Config(args[0]);
        System.out.println(config);
        check(config);



    }

    private static void check(Config config) throws Exception{
        String input = config.getString("input.folder");
        ClfDataSet dataSet = TRECFormat.loadClfDataSet(new File(input, "train.trec"),
                DataSetType.CLF_SPARSE, true);
        PriorProbClassifier priorProbClassifier = new PriorProbClassifier(dataSet.getNumClasses());
        priorProbClassifier.fit(dataSet);
        Map<String,Integer> ngramIndexMap = new HashMap<>();
        for (Feature feature: dataSet.getFeatureList().getAll()){
            int index = feature.getIndex();
            String ngram = ((Ngram)feature).getNgram();
            ngramIndexMap.put(ngram,index);
        }
        for (int k=0;k<dataSet.getNumClasses();k++){
            System.out.println("features for class "+k);
            List<Estimation> estimations = gradientSelection(config,dataSet,k,priorProbClassifier,ngramIndexMap);
            for (Estimation estimation: estimations){
                System.out.println(estimation);

            }



        }

    }

    private static double utility(ClfDataSet dataSet, int classIndex, int featureIndex, PriorProbClassifier priorProbClassifier){
        Vector vector = dataSet.getColumn(featureIndex);
        int[] labels = dataSet.getLabels();
        //actual and predicted
        double[] counts = new double[2];
        for (Vector.Element element: vector.nonZeroes()){
            int dataPoint = element.index();
            counts[1] += 1;
            if (labels[dataPoint]==classIndex){
                counts[0] += 1;
            }
        }
        return counts[0] - counts[1]*priorProbClassifier.getClassProbs()[classIndex];
    }

    //wierd
    private static Set<Integer> intersection(ClfDataSet dataSet, int featureIndex, Map<String, Integer> ngramIndexMap){
        String ngram = ((Ngram)dataSet.getFeatureList().get(featureIndex)).getNgram();
        String[] unigrams = ngram.split(" ");
        List<Set<Integer>> sets = Arrays.stream(unigrams).map(unigram -> {
            //todo why happen?
            if (!ngramIndexMap.containsKey(unigram)) {
                return new HashSet<Integer>();
            }
            int index = ngramIndexMap.get(unigram);
            Vector vector = dataSet.getColumn(index);
            Set<Integer> set = new HashSet<Integer>();
            for (Vector.Element element : vector.nonZeroes()) {
                set.add(element.index());
            }
            return set;
        }).collect(Collectors.toList());
        Set<Integer> intersection = sets.get(0);
        for (Set<Integer> set: sets){
            intersection.retainAll(set);
        }
        return intersection;
    }

    private static double lowerBound(ClfDataSet dataSet, int classIndex, int featureIndex, PriorProbClassifier priorProbClassifier, Map<String, Integer> ngramIndexMap){
        Vector vector = dataSet.getColumn(featureIndex);
        int[] labels = dataSet.getLabels();
        //actual and predicted
        double count = 0;
        for (Vector.Element element: vector.nonZeroes()){
            int dataPoint = element.index();

            if (labels[dataPoint]==classIndex){
                count += 1;
            }
        }
        double second = intersection(dataSet,featureIndex,ngramIndexMap).size() * priorProbClassifier.getClassProbs()[classIndex];
        return count - second;
    }

    private static double upperBound(ClfDataSet dataSet, int classIndex, int featureIndex, PriorProbClassifier priorProbClassifier, Map<String, Integer> ngramIndexMap){
        Set<Integer> intersection = intersection(dataSet,featureIndex,ngramIndexMap);
        Set<Integer> filtered = intersection.stream().filter(i -> dataSet.getLabels()[i]==classIndex).collect(Collectors.toSet());

        return filtered.size()*(1-priorProbClassifier.getClassProbs()[classIndex]);
    }

    private static List<Estimation> gradientSelection(Config config, ClfDataSet dataSet, int classIndex, PriorProbClassifier priorProbClassifier, Map<String, Integer> ngramIndexMap){
        int limit = config.getInt("limit");
        List<Estimation> top = dataSet.getFeatureList().getAll().parallelStream().map(feature -> {
            int index = feature.getIndex();
            double quality = utility(dataSet, classIndex, index, priorProbClassifier);
            Estimation estimation = new Estimation();
            estimation.setGradient(quality);
            double lower = lowerBound(dataSet, classIndex, index, priorProbClassifier, ngramIndexMap);
            double upper = upperBound(dataSet, classIndex, index, priorProbClassifier, ngramIndexMap);
            estimation.setLower(lower);
            estimation.setUpper(upper);
            String ngram = ((Ngram)feature).getNgram();
            estimation.setNgram(ngram);
            return estimation;
        }).sorted(Comparator.comparing(Estimation::getGradient).reversed())
                .limit(limit).collect(Collectors.toList());
        return top;
    }


    private static class Estimation{
        private String ngram;
        private double gradient;
        private double lower;
        private double upper;

        public double getGradient() {
            return gradient;
        }

        public void setGradient(double gradient) {
            this.gradient = gradient;
        }

        public double getLower() {
            return lower;
        }

        public void setLower(double lower) {
            this.lower = lower;
        }

        public double getUpper() {
            return upper;
        }

        public void setUpper(double upper) {
            this.upper = upper;
        }

        public String getNgram() {
            return ngram;
        }

        public void setNgram(String ngram) {
            this.ngram = ngram;
        }

        @Override
        public String toString() {
            final StringBuilder sb = new StringBuilder();
            sb.append(ngram).append('\t');
            sb.append(gradient).append('\t');
            sb.append(lower).append('\t');
            sb.append(upper).append('\t');
            return sb.toString();
        }
    }


}
