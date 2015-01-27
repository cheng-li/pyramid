package edu.neu.ccs.pyramid.feature_extraction;

import edu.neu.ccs.pyramid.dataset.DataSet;
import edu.neu.ccs.pyramid.dataset.DenseRegDataSet;
import edu.neu.ccs.pyramid.dataset.IdTranslator;
import edu.neu.ccs.pyramid.dataset.RegDataSetBuilder;
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
import java.util.concurrent.ConcurrentHashMap;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

/**
 * filter by minDf
 * rank by split
 * Created by chengli on 9/13/14.
 */
public class PhraseSplitExtractor {
    private static final Logger logger = LogManager.getLogger();
    private int minDf = 10;
    private ESIndex index;
    private int topN =20;
    private IdTranslator idTranslator;
    private int minDataPerLeaf = 2;
    private int lengthLimit = Integer.MAX_VALUE;

    public PhraseSplitExtractor(ESIndex index, IdTranslator idTranslator) {
        this.index = index;
        this.idTranslator = idTranslator;

    }

    public PhraseSplitExtractor setLengthLimit(int lengthLimit) {
        this.lengthLimit = lengthLimit;
        return this;
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
        System.out.println("gathering...");
        Collection<PhraseInfo> allPhrases = gather(focusSet,classIndex,seeds, validationSet);
        System.out.println("done");
        System.out.println("filtering...");
        List<PhraseInfo> candidates = filter(allPhrases,blacklist);
        System.out.println("done");
        System.out.println("ranking");
        List<String> ranked = rankBySplit(candidates,validationSet,residuals);
        System.out.println("done");
        return ranked;
    }

    List<PhraseInfo> getCandidates(FocusSet focusSet,int classIndex,
                               Set<String> seeds,List<Integer> validationSet,
                               Set<String> blacklist){
        System.out.println("gathering...");
        Collection<PhraseInfo> allPhrases = gather(focusSet,classIndex,seeds, validationSet);
        System.out.println("done");

        //todo change pos
//        System.out.println("gathered "+allPhrases.size() + " candidates");
        System.out.println("filtering...");
        List<PhraseInfo> candidates = filter(allPhrases,blacklist);
        System.out.println("done");
        return candidates;

    }

    public List<PhraseInfo> filter(Collection<PhraseInfo> phraseInfos, Set<String> blacklist){
        return phraseInfos.parallelStream()
                .filter(phraseInfo ->
                        ((phraseInfo.getSearchResponse().getHits().totalHits() > minDf)
                        && (!blacklist.contains(phraseInfo.getPhrase()))))
                .collect(Collectors.toList());
    }

    //todo
    private Collection<PhraseInfo> gather(FocusSet focusSet,
                                        int classIndex,
                                        Set<String> seeds,
                                        List<Integer> validationSet){
        String[] validationIndexIds = validationSet.parallelStream()
                .map(this.idTranslator::toExtId)
                .toArray(String[]::new);
        PhraseDetector phraseDetector = new PhraseDetector(index,validationIndexIds).setMinDf(this.minDf);
        phraseDetector.setLengthLimit(this.lengthLimit);

        List<Integer> dataPoints = focusSet.getDataClassK(classIndex);

        List<Set<PhraseInfo>> phrasesList = dataPoints.parallelStream()
                .map(dataPoint ->
                {
                    String indexId = idTranslator.toExtId(dataPoint);
                    Map<Integer, String> termVector = index.getTermVector(indexId);
                    Set<PhraseInfo> phraseInfos = phraseDetector.getPhraseInfos(termVector, seeds);
                    System.out.println("indexId = " + indexId + ", num ngrams = " + phraseInfos.size());
                    if (phraseInfos.size() > 1000) {
                        System.out.println(phraseInfos);
                    }

                    return phraseInfos;
                })
                .collect(Collectors.toList());

//        Set<PhraseInfo> all = new HashSet<>();
        Set<PhraseInfo> all = Collections.newSetFromMap(new ConcurrentHashMap<PhraseInfo, Boolean>());
        phrasesList.stream().parallel().forEach(all::addAll);
//        phrasesList.forEach(all::addAll);
        return all;
    }

    private List<String> rankBySplit(Collection<PhraseInfo> phraseInfos,
                                     List<Integer> validationSet,
                                     List<Double> residuals){
        //translate once
        String[] validationIndexIds = validationSet.parallelStream()
                .map(this.idTranslator::toExtId)
                .toArray(String[]::new);

        // this is stupid
        double[] residualsArray = residuals.stream().mapToDouble(a -> a).toArray();

        Comparator<Pair<PhraseInfo,Double>> pairComparator = Comparator.comparing(Pair::getSecond);
        List<String> goodPhrases = phraseInfos.stream().parallel()
                .map(phraseInfo ->
                        new Pair<>(phraseInfo, splitScore(phraseInfo, validationIndexIds, residualsArray)))
                .sorted(pairComparator.reversed())
                .map(pair -> pair.getFirst().getPhrase())
                .limit(this.topN)
                .collect(Collectors.toList());
        return goodPhrases;

    }

    double splitScore(PhraseInfo phraseInfo,
                              String[] validationSet,
                              double[] residuals){
        int numDataPoints = validationSet.length;
        DataSet dataSet = RegDataSetBuilder.getBuilder().numDataPoints(numDataPoints).numFeatures(1).dense(true).build();
        SearchResponse response = phraseInfo.getSearchResponse();
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
