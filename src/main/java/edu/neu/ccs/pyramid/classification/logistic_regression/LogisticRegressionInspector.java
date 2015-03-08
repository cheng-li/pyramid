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
    public static List<FeatureUtility> topFeatures(LogisticRegression logisticRegression,
                                                         int k){
        List<String> featureNames = logisticRegression.getFeatureList().getAll()
                .stream().map(Feature::getName).collect(Collectors.toList());
        Vector weights = logisticRegression.getWeights().getWeightsWithoutBiasForClass(k);
        Comparator<FeatureUtility> comparator = Comparator.comparing(FeatureUtility::getUtility);
        List<FeatureUtility> list = IntStream.range(0,weights.size())
                .mapToObj(i -> new FeatureUtility(i,featureNames.get(i)).setUtility(weights.get(i)))
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
        FeatureList featureList = logisticRegression.getFeatureList();
        Set<Integer> usedFeatures = new HashSet<>();
        for (int k=0;k<logisticRegression.getNumClasses();k++){
            Vector vector = logisticRegression.getWeights().getWeightsWithoutBiasForClass(k);
            for (Vector.Element element: vector.nonZeroes()){
                usedFeatures.add(element.index());
            }
        }
        List<Ngram> ngrams = usedFeatures.stream().map(featureList::get).filter(feature -> feature instanceof Ngram)
                .map(feature -> (Ngram)feature).collect(Collectors.toList());
        int maxLength = ngrams.stream().mapToInt(Ngram::getN).max().getAsInt();
        int[] numbers = new int[maxLength];
        ngrams.stream().forEach(ngram -> numbers[ngram.getN()-1] +=1);
        StringBuilder sb = new StringBuilder();
        for (int n=1;n<=maxLength;n++){
            sb.append("# "+n+"-grams = "+numbers[n-1]);
            sb.append("; ");
        }
        return sb.toString();

    }


}
