package edu.neu.ccs.pyramid.feature_extraction;

import edu.neu.ccs.pyramid.configuration.Config;
import edu.neu.ccs.pyramid.elasticsearch.ESIndex;
import edu.neu.ccs.pyramid.elasticsearch.SingleLabelIndex;

import java.util.*;
import java.util.concurrent.ConcurrentHashMap;
import java.util.stream.Collectors;

/**
 * naive enumeration of ngrams
 * Created by chengli on 1/15/15.
 */
public class NgramEnumerator {

    /**
     * gather ngrams with document frequency >= threshold
     * @param index
     * @param ids
     * @param n
     * @param minDf
     * @return
     * @throws Exception
     */
    public static List<String> gatherNgrams(ESIndex index,
                                     String[] ids, int n, int minDf) throws Exception{
        Map<String,Integer> counts = new ConcurrentHashMap<>();
        Arrays.stream(ids).parallel().forEach(id -> {
            Map<Integer,String> termVector = index.getTermVector(id);
            Map<String, Integer> localCount = NgramEnumerator.getNgramCounts(termVector,n);
            for (Map.Entry<String, Integer> entry: localCount.entrySet()){
                String ngram = entry.getKey();
                int oldCount = counts.getOrDefault(ngram,0);
                //document count += 1
                int newCount = oldCount + 1;
                counts.put(ngram,newCount);
            }
        });
        return counts.entrySet().parallelStream().filter(entry -> entry.getValue()>=minDf)
                .map(Map.Entry::getKey).collect(Collectors.toList());

    }

    /**
     * ngram counts in one doc
     * @param termVector
     * @param n
     * @return
     */
    public static Map<String, Integer> getNgramCounts(Map<Integer, String> termVector, int n){
        Map<String,Integer> counts = new HashMap<>();
        Comparator<Map.Entry<Integer,String>> comparator = Comparator.comparing(Map.Entry::getKey);
        List<Map.Entry<Integer,String>> entryList = termVector.entrySet().stream()
                .sorted(comparator).collect(Collectors.toList());
        if (entryList.size()==0){
            return counts;
        }

        List<String> sequence = new ArrayList<>();
        int last = -2;
        for (Map.Entry<Integer, String> entry: entryList){
            int index = entry.getKey();
            String term = entry.getValue();
            if (index!=last+1){
                updateNgramCounts(sequence,n,counts);
                sequence = new ArrayList<>();
            }
            last = index;
            sequence.add(term);
        }
        updateNgramCounts(sequence,n,counts);
        return counts;
    }

    /**
     * generate ngrams from contiguous sequence of strings
     * @param sequence
     * @return
     */
    static void updateNgramCounts(List<String> sequence, int n, Map<String, Integer> map){

        if (n>sequence.size()){
            return;
        }

        for (int start = 0;start<=sequence.size()-n;start++){
            String ngram = toNgram(sequence,start,start+n-1);
            int oldCount = map.getOrDefault(ngram,0);
            int newCount = oldCount + 1;
            map.put(ngram,newCount);
        }
    }

    /**
     *
     * @param sequence
     * @param start inclusive
     * @param end inclusive
     * @return
     */
     static String toNgram(List<String> sequence, int start, int end){
        String str = "";
        for (int i=start;i<end;i++){
            str = str.concat(sequence.get(i));
            str = str.concat(" ");
        }
        str = str.concat(sequence.get(end));
        return str;
    }
}
