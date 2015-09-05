package edu.neu.ccs.pyramid.experiment;

import edu.neu.ccs.pyramid.classification.PriorProbClassifier;
import edu.neu.ccs.pyramid.classification.logistic_regression.ElasticNetLogisticTrainer;
import edu.neu.ccs.pyramid.classification.logistic_regression.LogisticRegression;
import edu.neu.ccs.pyramid.classification.logistic_regression.LogisticRegressionInspector;
import edu.neu.ccs.pyramid.classification.logistic_regression.RidgeLogisticTrainer;
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
 * approximate logistic gradients
 * Created by chengli on 3/23/15.
 */
public class Exp75 {
    public static void main(String[] args) throws Exception{
        if (args.length !=1){
            throw new IllegalArgumentException("Please specify a properties file.");
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

        List<double[]> probs;
        probs = priorProbClassifier.predictClassProbs(dataSet);
        for (int k=0;k<dataSet.getNumClasses();k++){
            System.out.println("with prior probabilities:");
            System.out.println("features for class "+k);
            List<Estimation> estimations = gradientSelection(config,dataSet,k,probs,ngramIndexMap);
            for (Estimation estimation: estimations){
                if (estimation.getNgram().split(" ").length>1){
                    System.out.println(estimation);
                }
            }
        }
        LogisticRegression logisticRegression = new LogisticRegression(dataSet.getNumClasses(),dataSet.getNumFeatures());
        ElasticNetLogisticTrainer trainer = new ElasticNetLogisticTrainer.Builder(logisticRegression,dataSet)
                .setRegularization(0.00000001)
                .setL1Ratio(0).build();
        for (int i = 0;i<5;i++){
            trainer.iterate();
            probs = logisticRegression.predictClassProbs(dataSet);
            System.out.println("after iteration "+i);
            for (int k=0;k<dataSet.getNumClasses();k++){
                System.out.println("features for class "+k);
                List<Estimation> estimations = gradientSelection(config,dataSet,k,probs,ngramIndexMap);
                for (Estimation estimation: estimations){
                    if (estimation.getNgram().split(" ").length>1){
                        System.out.println(estimation);
                    }
                }
            }
        }

    }

    private static double utility(ClfDataSet dataSet, int classIndex, int featureIndex, List<double[]> probs){
        Vector vector = dataSet.getColumn(featureIndex);
        int[] labels = dataSet.getLabels();
        //actual and predicted
        double[] counts = new double[2];
        for (Vector.Element element: vector.nonZeroes()){
            int dataPoint = element.index();
            counts[1] += probs.get(dataPoint)[classIndex];
            if (labels[dataPoint]==classIndex){
                counts[0] += 1;
            }
        }
        return counts[0] - counts[1];
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

    private static double lowerBound(ClfDataSet dataSet, int classIndex, int featureIndex, List<double[]> probs, Map<String, Integer> ngramIndexMap){
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
        Set<Integer> intersection = intersection(dataSet,featureIndex,ngramIndexMap);
        double second = intersection.stream().mapToDouble(i -> probs.get(i)[classIndex]).sum();
        return count - second;
    }

    private static double upperBound(ClfDataSet dataSet, int classIndex, int featureIndex, List<double[]> probs, Map<String, Integer> ngramIndexMap){
        Set<Integer> intersection = intersection(dataSet,featureIndex,ngramIndexMap);
        Set<Integer> filtered = intersection.stream().filter(i -> dataSet.getLabels()[i]==classIndex).collect(Collectors.toSet());
        double term2 = filtered.stream().mapToDouble(i -> probs.get(i)[classIndex]).sum();
        return filtered.size() - term2;
    }


    private static double approximate(ClfDataSet dataSet, int classIndex, int featureIndex, List<double[]> probs, Map<String, Integer> ngramIndexMap){
        int[] labels = dataSet.getLabels();
        Set<Integer> intersection = intersection(dataSet,featureIndex,ngramIndexMap);
        Set<Integer> intersectionK = intersection.stream().filter(i -> labels[i]==classIndex).collect(Collectors.toSet());
        Set<Integer> intersectionNotK = intersection.stream().filter(i -> labels[i]!=classIndex).collect(Collectors.toSet());
        double ngramK = 0;
        double ngramNotK = 0;
        Vector vector = dataSet.getColumn(featureIndex);
        for (Vector.Element element: vector.nonZeroes()){
            int dataIndex = element.index();
            if (labels[dataIndex]==classIndex){
                ngramK += 1;
            } else {
                ngramNotK += 1;
            }
        }
        double term2 = intersectionK.stream().mapToDouble(i -> probs.get(i)[classIndex]).sum() * ngramK/intersectionK.size();
        double term3 = intersectionNotK.stream().mapToDouble(i -> probs.get(i)[classIndex]).sum() * ngramNotK/intersectionNotK.size();

        return ngramK - term2 - term3;
    }

    private static List<Estimation> gradientSelection(Config config, ClfDataSet dataSet, int classIndex, List<double[]> probs, Map<String, Integer> ngramIndexMap){
        int limit = config.getInt("limit");
        List<Estimation> top = dataSet.getFeatureList().getAll().parallelStream().map(feature -> {
            int index = feature.getIndex();
            double quality = utility(dataSet, classIndex, index, probs);
            Estimation estimation = new Estimation();
            estimation.setGradient(quality);
            double lower = lowerBound(dataSet, classIndex, index, probs, ngramIndexMap);
            double upper = upperBound(dataSet, classIndex, index, probs, ngramIndexMap);
            double approximate = approximate(dataSet, classIndex, index, probs, ngramIndexMap);
            estimation.setLower(lower);
            estimation.setUpper(upper);
            estimation.setApproximate(approximate);
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
        private double approximate;


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

        public double getApproximate() {
            return approximate;
        }

        public void setApproximate(double approximate) {
            this.approximate = approximate;
        }



        @Override
        public String toString() {
            final StringBuilder sb = new StringBuilder();
            sb.append(ngram).append("\t\t\t");
            sb.append(gradient).append('\t');
            sb.append(approximate).append('\t');
            sb.append(lower).append('\t');
            sb.append(upper).append('\t');
            return sb.toString();
        }
    }


}
