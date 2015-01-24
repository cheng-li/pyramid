package edu.neu.ccs.pyramid.feature_extraction;

import edu.neu.ccs.pyramid.dataset.DataSet;
import edu.neu.ccs.pyramid.dataset.IdTranslator;
import edu.neu.ccs.pyramid.dataset.RegDataSetBuilder;
import edu.neu.ccs.pyramid.elasticsearch.ESIndex;
import edu.neu.ccs.pyramid.regression.regression_tree.RegTreeConfig;
import edu.neu.ccs.pyramid.regression.regression_tree.RegTreeTrainer;
import edu.neu.ccs.pyramid.regression.regression_tree.RegressionTree;
import edu.neu.ccs.pyramid.util.Pair;
import org.apache.commons.lang3.time.StopWatch;
import org.elasticsearch.action.search.SearchResponse;
import org.elasticsearch.search.SearchHit;

import java.util.*;
import java.util.concurrent.ConcurrentHashMap;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

/**
 * enumerate all ngrams in the focus set and validation them on validation set
 * Created by chengli on 1/23/15.
 */
public class MixedBruteForceSplitExtractor {
    private ESIndex index;
    private IdTranslator idTranslator;
    private int minDataPerLeaf = 2;
    private List<Integer> ns;


    public MixedBruteForceSplitExtractor(ESIndex index, IdTranslator idTranslator) {
        this.index = index;
        this.idTranslator = idTranslator;
        this.ns = new ArrayList<>();
        ns.add(1);
        ns.add(2);
        ns.add(3);
    }

    public void setMinDataPerLeaf(int minDataPerLeaf) {
        this.minDataPerLeaf = minDataPerLeaf;
    }




    public List<String> gather(FocusSet focusSet,
                               int classIndex,
                               List<Integer> ns) throws Exception{
        List<Integer> dataPoints = focusSet.getDataClassK(classIndex);
        String[] indexIds = dataPoints.parallelStream()
                .map(idTranslator::toExtId).toArray(String[]::new);

        Set<String> allNgrams = new HashSet<>();
        for (int n: ns){
            List<String> ngrams = NgramEnumerator.gatherNgrams(this.index,indexIds,n,1);
            allNgrams.addAll(ngrams);
        }
        return new ArrayList<>(allNgrams);
    }

    public List<String> filter(Collection<String> candidates, Set<String> blacklist){
        return candidates.stream().parallel().filter(candidate -> !blacklist.contains(candidate))
                .collect(Collectors.toList());
    }

    double splitScore(String phrase,
                      String[] validationSet,
                      double[] residuals){
        PhraseInfo phraseInfo = new PhraseInfo(phrase);
        SearchResponse searchResponse = index.matchPhrase(index.getBodyField(),
                phrase,validationSet,0);
        phraseInfo.setSearchResponse(searchResponse);

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

    private List<String> rankBySplit(Collection<String> phrases,
                                     List<Integer> validationSet,
                                     List<Double> residuals,
                                     int topN){
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
                .map(pair -> pair.getFirst())
                .limit(topN)
                .collect(Collectors.toList());
        return goodPhrases;

    }

    public List<String> getGoodPhrases(FocusSet focusSet,
                                       List<Integer> validationSet,
                                       Set<String> blacklist,
                                       int classIndex,
                                       List<Double> residuals,
                                       int topN) throws Exception{


        List<String> goodPhrases = new ArrayList<String>();
        if (topN==0){
            return goodPhrases;
        }
        System.out.println("gathering...");
        Collection<String> allPhrases = gather(focusSet,classIndex,ns);
        System.out.println("done");
        System.out.println("filtering...");
        List<String> candidates = filter(allPhrases,blacklist);
        System.out.println("done");
        System.out.println("ranking");
        List<String> ranked = rankBySplit(candidates,validationSet,residuals,topN);
        System.out.println("done");
        return ranked;
    }

}
