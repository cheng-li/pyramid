package edu.neu.ccs.pyramid.classification.logistic_regression;

import edu.neu.ccs.pyramid.feature.Feature;
import edu.neu.ccs.pyramid.feature.FeatureList;
import edu.neu.ccs.pyramid.feature.FeatureUtility;
import edu.neu.ccs.pyramid.feature.Ngram;
import org.apache.mahout.math.Vector;

import java.util.Comparator;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import java.util.stream.Collectors;
import java.util.stream.IntStream;


/**
 * Created by chengli on 12/7/14.
 */
public class LogisticRegressionInspector {
    //todo if featureList are on different scales, weights are not comparable

    /**
     * unused features won't be considered
     * @param logisticRegression
     * @param k
     * @return
     */
    public static List<FeatureUtility> topFeatures(LogisticRegression logisticRegression,
                                                         int k){
        FeatureList featureList = logisticRegression.getFeatureList();
        Vector weights = logisticRegression.getWeights().getWeightsWithoutBiasForClass(k);
        Comparator<FeatureUtility> comparator = Comparator.comparing(FeatureUtility::getUtility);
        List<FeatureUtility> list = IntStream.range(0,weights.size())
                .mapToObj(i -> new FeatureUtility(featureList.get(i)).setUtility(weights.get(i)))
                .filter(featureUtility -> featureUtility.getUtility()>0)
                .sorted(comparator.reversed())
                .collect(Collectors.toList());
        IntStream.range(0,list.size()).forEach(i-> list.get(i).setRank(i));
        return list;
    }

    public static List<FeatureUtility> topFeatures(LogisticRegression logisticRegression,
                                                         int k,
                                                         int limit){
        return topFeatures(logisticRegression,k).stream().limit(limit).collect(Collectors.toList());
    }

    public static int[] numOfUsedFeaturesEachClass(LogisticRegression logisticRegression){
        int[] numbers = new int[logisticRegression.getNumClasses()];
        for (int k=0;k<logisticRegression.getNumClasses();k++){
            numbers[k] = logisticRegression.getWeights().getWeightsWithoutBiasForClass(k).getNumNonZeroElements();
        }
        return numbers;
    }

    public static int numOfUsedFeaturesCombined(LogisticRegression logisticRegression){
        Set<Integer> usedFeatures = new HashSet<>();
        for (int k=0;k<logisticRegression.getNumClasses();k++){
            Vector vector = logisticRegression.getWeights().getWeightsWithoutBiasForClass(k);
            for (Vector.Element element: vector.nonZeroes()){
                usedFeatures.add(element.index());
            }
        }
        return usedFeatures.size();

    }

    public static String checkNgramUsage(LogisticRegression logisticRegression){
        StringBuilder sb = new StringBuilder();
        FeatureList featureList = logisticRegression.getFeatureList();
        Set<Integer> usedFeatures = new HashSet<>();
        for (int k=0;k<logisticRegression.getNumClasses();k++){
            Vector vector = logisticRegression.getWeights().getWeightsWithoutBiasForClass(k);
            for (Vector.Element element: vector.nonZeroes()){
                usedFeatures.add(element.index());
            }
        }

        List<Ngram> selected = usedFeatures.stream().map(featureList::get).filter(feature -> feature instanceof Ngram)
                .map(feature -> (Ngram)feature).collect(Collectors.toList());

        List<Ngram> candidates = featureList.getAll().stream()
                .filter(feature -> feature instanceof Ngram)
                .map(feature -> (Ngram)feature).collect(Collectors.toList());
        int maxLength = candidates.stream().mapToInt(Ngram::getN).max().getAsInt();
        int[] numberCandidates = new int[maxLength];
        candidates.stream().forEach(ngram -> numberCandidates[ngram.getN() - 1] += 1);
        sb.append("number of ngram candidates: ");
        for (int n=1;n<=maxLength;n++){
            sb.append(n+"-grams = "+numberCandidates[n-1]);
            sb.append("; ");
        }
        sb.append("\n");

        int[] numberSelected = new int[maxLength];
        selected.stream().forEach(ngram -> numberSelected[ngram.getN() - 1] += 1);
        sb.append("number of selected ngram: ");
        for (int n=1;n<=maxLength;n++){
            sb.append(+n+"-grams = "+numberSelected[n-1]);
            sb.append("; ");
        }
        sb.append("\n");

        int[] easyCandidates = new int[maxLength];
        int[] easySelected = new int[maxLength];
        Set<String> unigrams = selected.stream().filter(ngram -> ngram.getN() == 1)
                .map(Ngram::getNgram).collect(Collectors.toSet());

        candidates.stream().filter(ngram -> isComposedOf(ngram.getNgram(), unigrams))
                .forEach(ngram -> easyCandidates[ngram.getN() - 1] += 1);
        sb.append("number of ngram candidates that can be constructed from seeds: ");
        for (int n=1;n<=maxLength;n++){
            sb.append(n+"-grams = "+easyCandidates[n-1]);
            sb.append("; ");
        }
        sb.append("\n");

        selected.stream().filter(ngram -> isComposedOf(ngram.getNgram(), unigrams))
                .forEach(ngram -> easySelected[ngram.getN() - 1] += 1);
        sb.append("number of selected ngrams that can be constructed from seeds: ");
        for (int n=1;n<=maxLength;n++){
            sb.append(n+"-grams = "+easySelected[n-1]);
            sb.append("; ");
        }
        sb.append("\n");

        sb.append("percentage of selected ngrams that can be constructed from seeds: ");
        for (int n=1;n<=maxLength;n++){
            sb.append(n+"-grams = "+(double)easySelected[n-1]/numberSelected[n-1]);
            sb.append("; ");
        }
        sb.append("\n");

        sb.append("feature selection ratio: ");
        for (int n=1;n<=maxLength;n++){
            sb.append(n+"-grams = "+(double)numberSelected[n-1]/numberCandidates[n-1]);
            sb.append("; ");
        }
        sb.append("\n");

        sb.append("feature selection ratio based on seeds: ");
        for (int n=1;n<=maxLength;n++){
            sb.append(n+"-grams = "+(double)easySelected[n-1]/easyCandidates[n-1]);
            sb.append("; ");
        }

        return sb.toString();

    }

    private static boolean isComposedOf(String ngram, Set<String> unigrams){
        String[] split = ngram.split(" ");
        for (String term: split){
            if (unigrams.contains(term)){
                return true;
            }
        }
        return false;
    }


}
