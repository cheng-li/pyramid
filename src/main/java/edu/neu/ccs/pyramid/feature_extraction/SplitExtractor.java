package edu.neu.ccs.pyramid.feature_extraction;

import edu.neu.ccs.pyramid.dataset.DataSet;
import edu.neu.ccs.pyramid.dataset.DenseRegDataSet;
import edu.neu.ccs.pyramid.elasticsearch.ESIndex;
import edu.neu.ccs.pyramid.elasticsearch.IdTranslator;
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
 * consider all terms in the focus set
 * for each term, build a regression stump for the validation set
 * return terms with highest variance reduction
 * Created by chengli on 9/7/14.
 */
public class SplitExtractor {
    private static final Logger logger = LogManager.getLogger();
    private ESIndex index;
    /**
     * max number of good ngrams to return for each class
     */
    //TODO: different for different class
    private int topN;
    private int minDataPerLeaf=1;
    private IdTranslator idTranslator;

    public SplitExtractor(ESIndex index, IdTranslator idTranslator,
                          int topN) {
        this.index = index;
        this.topN = topN;
        this.idTranslator = idTranslator;
    }

    public SplitExtractor setMinDataPerLeaf(int minDataPerLeaf) {
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

        Collection<String> terms = gather(focusSet,classIndex,blacklist);
        return rankBySplit(terms,validationSet,residuals);
    }


    /**
     * don't need to fetch stats information
     * @param focusSet
     * @param classIndex
     * @return
     */
    private Collection<String> gather(FocusSet focusSet,
                                        int classIndex,
                                        Set<String> blacklist){
        List<Integer> dataPoints = focusSet.getDataClassK(classIndex);
        //we don't union sets as we need to combine stats
        List<Set<String>> termSetList = dataPoints.parallelStream()
                .map(dataPoint ->
                {String indexId = idTranslator.toIndexId(dataPoint);
                    Set<String> terms = null;
                    try {
                        terms = index.getTerms(indexId);
                    } catch (IOException e) {
                        e.printStackTrace();
                    }
                    return terms;})
                .collect(Collectors.toList());

        //it is easier to do this in a single thread
        Set<String> allTerms = new HashSet<>();
        termSetList.forEach(allTerms::addAll);
        allTerms.removeAll(blacklist);
        return allTerms;
    }

    private List<String> rankBySplit(Collection<String> terms,
                                     List<Integer> validationSet,
                                     List<Double> residuals){
        //translate once
        String[] validationIndexIds = validationSet.parallelStream()
                .map(this.idTranslator::toIndexId)
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
            double value = matchingScores.getOrDefault(validationSet[i],0f);
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
