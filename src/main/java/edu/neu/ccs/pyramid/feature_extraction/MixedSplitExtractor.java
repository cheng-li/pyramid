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



    public MixedSplitExtractor(TermTfidfSplitExtractor termTfidfSplitExtractor,
                               PhraseSplitExtractor phraseSplitExtractor) {
        this.termTfidfSplitExtractor = termTfidfSplitExtractor;
        this.phraseSplitExtractor = phraseSplitExtractor;
    }

//    public List<String> getGoodNgrams(FocusSet focusSet,
//                                       Set<String> blacklist,
//                                       int classIndex,
//                                       List<Double> residuals,
//                                       Set<String> seeds) throws Exception{
//        List<String> termCandidates = termTfidfSplitExtractor.getCandidates(focusSet,classIndex,blacklist);
//
//        Map<String, Double> scores = new ConcurrentHashMap<>();
//
//
//        double[] residualsArray = residuals.stream().mapToDouble(a -> a).toArray();
//        termCandidates.stream().parallel()
//                .forEach(term ->
//                        scores.put(term, termTfidfSplitExtractor.splitScore(term, residualsArray)));
//
//        List<PhraseInfo> phraseCandidates = phraseSplitExtractor.getCandidates(focusSet,classIndex,seeds,blacklist);
//        phraseCandidates.stream().parallel()
//                .forEach(phraseInfo ->
//                        scores.put(phraseInfo.getPhrase(), phraseSplitExtractor.splitScore(phraseInfo, residualsArray)));
//
//
//        Comparator<Map.Entry<String,Double>> comparator = Comparator.comparing(Map.Entry::getValue);
//        return scores.entrySet().stream().parallel().sorted(comparator.reversed()).limit(topN)
//                .map(Map.Entry::getKey).collect(Collectors.toList());
//
//
//    }

    public List<String> getGoodNgrams(FocusSet focusSet,
                                      Set<String> blacklist,
                                      int classIndex,
                                      List<Double> residuals,
                                      int numSeeds,
                                      int topN) throws Exception{

        // don't want the blacklist to impact seeds
        List<String> termCandidates = termTfidfSplitExtractor.getCandidates(focusSet,classIndex);

        Map<String, Double> scores = new ConcurrentHashMap<>();


        double[] residualsArray = residuals.stream().mapToDouble(a -> a).toArray();
        termCandidates.stream().parallel()
                .forEach(term ->
                        scores.put(term, termTfidfSplitExtractor.splitScore(term, residualsArray)));

        Comparator<Map.Entry<String,Double>> comparator = Comparator.comparing(Map.Entry::getValue);
        Set<String> seeds = scores.entrySet().stream().sorted(comparator.reversed()).limit(numSeeds)
                .map(Map.Entry::getKey).collect(Collectors.toSet());
        System.out.println("seeds = "+seeds);

        System.out.println("get ngram candidates...");
        List<PhraseInfo> phraseCandidates = phraseSplitExtractor.getCandidates(focusSet,classIndex,seeds,blacklist);
        System.out.println("done");
        System.out.println("calculating ngram validation scores...");
        phraseCandidates.stream().parallel()
                .forEach(phraseInfo ->
                        scores.put(phraseInfo.getPhrase(), phraseSplitExtractor.splitScore(phraseInfo, residualsArray)));

        System.out.println("done");

        int totalCandidates = termCandidates.size()+ phraseCandidates.size();
        System.out.println("number of ngram candidates for validation = "+totalCandidates);

        return scores.entrySet().stream().parallel().sorted(comparator.reversed())
                .map(Map.Entry::getKey).filter(key -> !blacklist.contains(key))
                .limit(topN).collect(Collectors.toList());


    }
}
