package edu.neu.ccs.pyramid.feature_extraction;


import java.util.*;
import java.util.concurrent.ConcurrentHashMap;
import java.util.stream.Collectors;

/**
 * rank unigrams and ngrams together
 * Created by chengli on 12/29/14.
 */
public class MixedSplitExtractor {
    private TermTfidfSplitExtractor termTfidfSplitExtractor;
    private PhraseSplitExtractor phraseSplitExtractor;
    private int topN =20;


    public void setTopN(int topN) {
        this.topN = topN;
    }

    public MixedSplitExtractor(TermTfidfSplitExtractor termTfidfSplitExtractor,
                               PhraseSplitExtractor phraseSplitExtractor) {
        this.termTfidfSplitExtractor = termTfidfSplitExtractor;
        this.phraseSplitExtractor = phraseSplitExtractor;
    }

    public List<String> getGoodNgrams(FocusSet focusSet,
                                       List<Integer> validationSet,
                                       Set<String> blacklist,
                                       int classIndex,
                                       List<Double> residuals,
                                       Set<String> seeds) throws Exception{
        List<String> termCandidates = termTfidfSplitExtractor.getCandidates(focusSet,classIndex,blacklist);
        List<PhraseInfo> phraseCandidates = phraseSplitExtractor.getCandidates(focusSet,classIndex,seeds,validationSet,blacklist);
        Map<String, Double> scores = new ConcurrentHashMap<>();

        String[] validationIndexIds = validationSet.parallelStream()
                .map(termTfidfSplitExtractor.idTranslator::toExtId)
                .toArray(String[]::new);

        double[] residualsArray = residuals.stream().mapToDouble(a -> a).toArray();
        termCandidates.stream().parallel()
                .forEach(term ->
                        scores.put(term, termTfidfSplitExtractor.splitScore(term, validationIndexIds, residualsArray)));


        phraseCandidates.stream().parallel()
                .forEach(phraseInfo ->
                        scores.put(phraseInfo.getPhrase(), phraseSplitExtractor.splitScore(phraseInfo, validationIndexIds, residualsArray)));


        Comparator<Map.Entry<String,Double>> comparator = Comparator.comparing(Map.Entry::getValue);
        return scores.entrySet().stream().sorted(comparator.reversed()).limit(topN)
                .map(Map.Entry::getKey).collect(Collectors.toList());


    }
}
