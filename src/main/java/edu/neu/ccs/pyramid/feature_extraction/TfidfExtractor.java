package edu.neu.ccs.pyramid.feature_extraction;

import edu.neu.ccs.pyramid.elasticsearch.ESIndex;
import edu.neu.ccs.pyramid.dataset.IdTranslator;
import edu.neu.ccs.pyramid.elasticsearch.TermStat;
import org.apache.commons.lang3.time.StopWatch;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

import java.io.IOException;
import java.util.*;
import java.util.stream.Collectors;

/**
 * filter all terms in the focus set by minDf
 * sort passed terms by tfidf
 * return top tfidf terms
 * Created by chengli on 9/6/14.
 */
public class TfidfExtractor {
    private static final Logger logger = LogManager.getLogger();
    private ESIndex index;
    /**
     * max number of good terms to return for each class
     */
    //TODO: different for different class
    private int topN;
    private IdTranslator idTranslator;
    /**
     * in the whole collection
     */
    private int minDf=1;


    public TfidfExtractor(ESIndex index,
                          IdTranslator idTranslator,
                          int topN) {
        this.index = index;
        this.idTranslator = idTranslator;
        this.topN = topN;
    }

    public TfidfExtractor setMinDf(int minDf) {
        this.minDf = minDf;
        return this;
    }

    /**
     * multi-threaded
     * @param focusSet
     * @param  blacklist
     * @param classIndex class index
     * @return good terms for one class
     */
    public List<String> getGoodTerms(FocusSet focusSet,
                                     Set<String> blacklist,
                                     int classIndex) throws Exception{
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
        goodTerms = filter(termStats,blacklist);

        return goodTerms;
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
                .sorted(comparator.reversed()).limit(this.topN)
                .map(TermStat::getTerm).collect(Collectors.toList());
        return terms;
    }
}
