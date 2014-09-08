package edu.neu.ccs.pyramid.feature_extraction;

import edu.neu.ccs.pyramid.dataset.DataSet;
import edu.neu.ccs.pyramid.dataset.DenseRegDataSet;
import edu.neu.ccs.pyramid.elasticsearch.ESIndex;
import edu.neu.ccs.pyramid.dataset.IdTranslator;
import edu.neu.ccs.pyramid.elasticsearch.TermStat;
import edu.neu.ccs.pyramid.regression.regression_tree.RegTreeConfig;
import edu.neu.ccs.pyramid.regression.regression_tree.RegTreeTrainer;
import edu.neu.ccs.pyramid.regression.regression_tree.RegressionTree;
import edu.neu.ccs.pyramid.util.Pair;
import org.apache.commons.lang3.time.StopWatch;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.elasticsearch.action.search.SearchResponse;
import org.elasticsearch.index.query.MatchQueryBuilder;
import org.elasticsearch.search.SearchHit;

import java.io.IOException;
import java.util.*;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

/**
 * filter all ngrams in the focus set by minDf
 * sort passed ngrams by tfidf
 * keep high tf-idf ngrams
 * for each high tf-idf ngram, build a regression stump for the validation set
 * return ngrams with highest variance reduction
 * Created by chengli on 9/6/14.
 */
public class TfidfSplitExtractor {
    private static final Logger logger = LogManager.getLogger();
    private ESIndex index;
    /**
     * max number of good ngrams to return for each class
     */
    //TODO: different for different class
    private int topN;
    private IdTranslator idTranslator;
    /**
     * in the whole collection
     */
    private int minDf=1;
    /**
     * how many terms can pass the tf-idf filter?
     */
    private int numSurvivors=50;

    /**
     * regression tree on the validation set
     */
    int minDataPerLeaf=1;

    public TfidfSplitExtractor(ESIndex index,
                               IdTranslator idTranslator,
                               int topN) {
        this.index = index;
        this.idTranslator = idTranslator;
        this.topN = topN;
    }

    public TfidfSplitExtractor setMinDf(int minDf) {
        this.minDf = minDf;
        return this;
    }

    public TfidfSplitExtractor setNumSurvivors(int numSurvivors) {
        this.numSurvivors = numSurvivors;
        return this;
    }

    public TfidfSplitExtractor setMinDataPerLeaf(int minDataPerLeaf) {
        this.minDataPerLeaf = minDataPerLeaf;
        return this;
    }

    /**
     *
     * @param focusSet
     * @param validationSet algorithm ids
     * @param blacklist
     * @param classIndex
     * @param residuals  residuals of calidationSet, column vector
     * @return
     * @throws Exception
     */
    public List<String> getGoodTerms(FocusSet focusSet,
                                     List<Integer> validationSet,
                                     Set<String> blacklist,
                                     int classIndex,
                                     List<Double> residuals) throws Exception{
        StopWatch stopWatch = null;
        if (logger.isDebugEnabled()){
            stopWatch = new StopWatch();
            stopWatch.start();
        }
        List<String> goodTerms = new ArrayList<String>();
        if (this.topN==0){
            return goodTerms;
        }

        Collection<TermStat> termStats = gather(focusSet,classIndex);
        List<String> termCandidates = filter(termStats,blacklist);
        return rankBySplit(termCandidates,validationSet,residuals);
    }


    /**
     * gather term stats from focus set
     * @param focusSet
     * @param classIndex
     * @return
     */
    private Collection<TermStat> gather(FocusSet focusSet,
                                        int classIndex){
        List<Integer> dataPoints = focusSet.getDataClassK(classIndex);
        //we don't union sets as we need to combine stats
        List<Set<TermStat>> termStatSetList = dataPoints.parallelStream()
                .map(dataPoint ->
                {String indexId = idTranslator.toExtId(dataPoint);
                    Set<TermStat> termStats = null;
                    try {
                        termStats = index.getTermStats(indexId);
                    } catch (IOException e) {
                        e.printStackTrace();
                    }
                    return termStats;})
                .collect(Collectors.toList());

        //it is easier to do this in a single thread
        Map<String, TermStat> termMap = new HashMap<>();
        for (Set<TermStat> set: termStatSetList){
            for (TermStat termStat: set){
                String term = termStat.getTerm();
                if (termMap.containsKey(term)){
                    //combines tfidf for multiple appearance of the same term
                    TermStat oldStat = termMap.get(term);
                    TermStat combined = TermStat.combine(oldStat,termStat);
                    termMap.put(term, combined);
                } else {
                    termMap.put(term, termStat);
                }
            }
        }

        Collection<TermStat> termStats = termMap.values();
        return termStats;
    }
    /**
     * filter by minDf, sort by tfidf
     * @return
     */
    private List<String> filter(Collection<TermStat> termStats,
                                Set<String> blacklist){
        Comparator<TermStat> comparator = Comparator.comparing(TermStat::getTfidf);

        List<String> terms = termStats.stream().parallel().
                filter(termStat -> (termStat.getDf()>=this.minDf)
                        &&(!blacklist.contains(termStat.getTerm())))
                .sorted(comparator.reversed()).limit(this.numSurvivors)
                .map(TermStat::getTerm).collect(Collectors.toList());
        return terms;
    }

    private List<String> rankBySplit(Collection<String> terms,
                                     List<Integer> validationSet,
                                     List<Double> residuals){
        //translate once
        String[] validationIndexIds = validationSet.parallelStream()
                .map(this.idTranslator::toExtId)
                .toArray(String[]::new);

        // this is stupid
        double[] residualsArray = residuals.stream().mapToDouble(a -> a).toArray();

        Comparator<Pair<String,Double>> pairComparator = Comparator.comparing(Pair::getSecond);
        List<String> goodTerms = terms.stream().parallel()
                .map(term ->
                        new Pair<>(term, splitScore(term, validationIndexIds, residualsArray)))
                .sorted(pairComparator.reversed())
                .map(Pair::getFirst)
                .limit(this.topN)
                .collect(Collectors.toList());
        return goodTerms;

    }

    /**
     * use matching scores as feature values
     * @param term
     * @param validationSet
     * @param residuals
     * @return
     */
    private double splitScore(String term,
                              String[] validationSet,
                              double[] residuals){
        int numDataPoints = validationSet.length;
        DataSet dataSet = new DenseRegDataSet(numDataPoints,1);
        SearchResponse response = this.index.match(this.index.getBodyField(),
                term,validationSet, MatchQueryBuilder.Operator.AND);
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
                .setActiveDataPoints(IntStream.range(0,validationSet.length).toArray())
                .setActiveFeatures(activeFeatures)
                .setMaxNumLeaves(2)
                .setMinDataPerLeaf(this.minDataPerLeaf);
        RegressionTree tree = RegTreeTrainer.fit(regTreeConfig,dataSet,residuals);
        return tree.getRoot().getReduction();
    }
}
