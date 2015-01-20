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

        Map<String, Double> scores = new ConcurrentHashMap<>();

        String[] validationIndexIds = validationSet.parallelStream()
                .map(termTfidfSplitExtractor.idTranslator::toExtId)
                .toArray(String[]::new);

        double[] residualsArray = residuals.stream().mapToDouble(a -> a).toArray();
        termCandidates.stream().parallel()
                .forEach(term ->
                        scores.put(term, termTfidfSplitExtractor.splitScore(term, validationIndexIds, residualsArray)));

        List<PhraseInfo> phraseCandidates = phraseSplitExtractor.getCandidates(focusSet,classIndex,seeds,validationSet,blacklist);
        phraseCandidates.stream().parallel()
                .forEach(phraseInfo ->
                        scores.put(phraseInfo.getPhrase(), phraseSplitExtractor.splitScore(phraseInfo, validationIndexIds, residualsArray)));


        Comparator<Map.Entry<String,Double>> comparator = Comparator.comparing(Map.Entry::getValue);
        return scores.entrySet().stream().sorted(comparator.reversed()).limit(topN)
                .map(Map.Entry::getKey).collect(Collectors.toList());


    }

    public List<String> getGoodNgrams(FocusSet focusSet,
                                      List<Integer> validationSet,
                                      Set<String> blacklist,
                                      int classIndex,
                                      List<Double> residuals,
                                      int numSeeds,
                                      int topN) throws Exception{

        // don't want the blacklist to impact seeds
        List<String> termCandidates = termTfidfSplitExtractor.getCandidates(focusSet,classIndex);

        Map<String, Double> scores = new ConcurrentHashMap<>();

        String[] validationIndexIds = validationSet.parallelStream()
                .map(termTfidfSplitExtractor.idTranslator::toExtId)
                .toArray(String[]::new);

        double[] residualsArray = residuals.stream().mapToDouble(a -> a).toArray();
        termCandidates.stream().parallel()
                .forEach(term ->
                        scores.put(term, termTfidfSplitExtractor.splitScore(term, validationIndexIds, residualsArray)));

        Comparator<Map.Entry<String,Double>> comparator = Comparator.comparing(Map.Entry::getValue);
        Set<String> seeds = scores.entrySet().stream().sorted(comparator.reversed()).limit(numSeeds)
                .map(Map.Entry::getKey).collect(Collectors.toSet());
        System.out.println("seeds = "+seeds);

        List<PhraseInfo> phraseCandidates = phraseSplitExtractor.getCandidates(focusSet,classIndex,seeds,validationSet,blacklist);
        phraseCandidates.stream().parallel()
                .forEach(phraseInfo ->
                        scores.put(phraseInfo.getPhrase(), phraseSplitExtractor.splitScore(phraseInfo, validationIndexIds, residualsArray)));



        return scores.entrySet().stream().sorted(comparator.reversed())
                .map(Map.Entry::getKey).filter(key->!blacklist.contains(key))
                .limit(topN).collect(Collectors.toList());


    }
}
