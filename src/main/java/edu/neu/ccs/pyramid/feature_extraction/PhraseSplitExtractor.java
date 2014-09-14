package edu.neu.ccs.pyramid.feature_extraction;

import edu.neu.ccs.pyramid.dataset.DataSet;
import edu.neu.ccs.pyramid.dataset.DenseRegDataSet;
import edu.neu.ccs.pyramid.dataset.IdTranslator;
import edu.neu.ccs.pyramid.elasticsearch.ESIndex;
import edu.neu.ccs.pyramid.regression.regression_tree.RegTreeConfig;
import edu.neu.ccs.pyramid.regression.regression_tree.RegTreeTrainer;
import edu.neu.ccs.pyramid.regression.regression_tree.RegressionTree;
import edu.neu.ccs.pyramid.util.Pair;
import org.apache.commons.lang3.time.StopWatch;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.elasticsearch.action.search.SearchResponse;

import org.elasticsearch.search.SearchHit;


import java.util.*;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

/**
 * filter by minDf
 * rank by split
 * Created by chengli on 9/13/14.
 */
public class PhraseSplitExtractor {
    private static final Logger logger = LogManager.getLogger();
    int minDf = 10;
    private ESIndex index;
    private int topN =20;
    private IdTranslator idTranslator;
    int minDataPerLeaf = 2;

    public PhraseSplitExtractor(ESIndex index, IdTranslator idTranslator) {
        this.index = index;
        this.idTranslator = idTranslator;
    }

    public PhraseSplitExtractor setTopN(int topN) {
        this.topN = topN;
        return this;
    }

    public PhraseSplitExtractor setMinDf(int minDf) {
        this.minDf = minDf;
        return this;
    }

    public PhraseSplitExtractor setMinDataPerLeaf(int minDataPerLeaf) {
        this.minDataPerLeaf = minDataPerLeaf;
        return this;
    }

    public List<String> getGoodPhrases(FocusSet focusSet,
                                     List<Integer> validationSet,
                                     Set<String> blacklist,
                                     int classIndex,
                                     List<Double> residuals,
                                     Set<String> seeds) throws Exception{
        StopWatch stopWatch = null;
        if (logger.isDebugEnabled()){
            stopWatch = new StopWatch();
            stopWatch.start();
        }
        List<String> goodPhrases = new ArrayList<String>();
        if (this.topN==0){
            return goodPhrases;
        }

        Collection<String> allPhrases = gather(focusSet,classIndex,seeds);
        List<String> candidates = filter(allPhrases,blacklist);
        return rankBySplit(candidates,validationSet,residuals);
    }

    public List<String> filter(Collection<String> phrases,Set<String> blacklist){
        return phrases.parallelStream()
                .filter(phrase -> (index.phraseDF(index.getBodyField(), phrase, 0) > minDf)
                        && (!blacklist.contains(phrase)))
                .collect(Collectors.toList());
    }

    private Collection<String> gather(FocusSet focusSet,
                                        int classIndex,
                                        Set<String> seeds){
        List<Integer> dataPoints = focusSet.getDataClassK(classIndex);

        List<Set<String>> phrasesList = dataPoints.parallelStream()
                .map(dataPoint ->
                {String indexId = idTranslator.toExtId(dataPoint);
                  Map<Integer,String> termVector = index.getTermVector(indexId);
                    return PhraseDetector.getPhrases(termVector,seeds);})
                .collect(Collectors.toList());

        Set<String> all = new HashSet<>();
        phrasesList.forEach(all::addAll);
        return all;
    }

    private List<String> rankBySplit(Collection<String> phrases,
                                     List<Integer> validationSet,
                                     List<Double> residuals){
        //translate once
        String[] validationIndexIds = validationSet.parallelStream()
                .map(this.idTranslator::toExtId)
                .toArray(String[]::new);

        // this is stupid
        double[] residualsArray = residuals.stream().mapToDouble(a -> a).toArray();

        Comparator<Pair<String,Double>> pairComparator = Comparator.comparing(Pair::getSecond);
        List<String> goodPhrases = phrases.stream().parallel()
                .map(phrase ->
                        new Pair<>(phrase, splitScore(phrase, validationIndexIds, residualsArray)))
                .sorted(pairComparator.reversed())
                .map(Pair::getFirst)
                .limit(this.topN)
                .collect(Collectors.toList());
        return goodPhrases;

    }

    private double splitScore(String phrase,
                              String[] validationSet,
                              double[] residuals){
        int numDataPoints = validationSet.length;
        DataSet dataSet = new DenseRegDataSet(numDataPoints,1);
        SearchResponse response = this.index.matchPhrase(this.index.getBodyField(),
                phrase, validationSet, 0);
        Map<String,Float> matchingScores = new HashMap<>();
        for (SearchHit hit: response.getHits().getHits()){
            String indexId = hit.getId();
            float matchingScore = hit.getScore();
            matchingScores.put(indexId,matchingScore);
        }
        for (int i=0;i<numDataPoints;i++){
            double value = matchingScores.getOrDefault(validationSet[i], 0f);
            dataSet.setFeatureValue(i,0,value);
        }

        int[] activeFeatures = {0};
        RegTreeConfig regTreeConfig = new RegTreeConfig()
                .setActiveDataPoints(IntStream.range(0, validationSet.length).toArray())
                .setActiveFeatures(activeFeatures)
                .setMaxNumLeaves(2)
                .setMinDataPerLeaf(this.minDataPerLeaf);
        RegressionTree tree = RegTreeTrainer.fit(regTreeConfig, dataSet, residuals);
        return tree.getRoot().getReduction();
    }


}
