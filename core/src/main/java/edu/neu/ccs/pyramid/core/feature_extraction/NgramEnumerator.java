package edu.neu.ccs.pyramid.core.feature_extraction;

import com.google.common.collect.ConcurrentHashMultiset;
import com.google.common.collect.Multiset;
import edu.neu.ccs.pyramid.core.elasticsearch.ESIndex;
import edu.neu.ccs.pyramid.core.feature.Ngram;

import java.util.*;
import java.util.concurrent.ConcurrentHashMap;
import java.util.stream.Collectors;

/**
 * naive enumeration of ngrams
 * Created by chengli on 1/15/15.
 */
public class NgramEnumerator {

    public static Multiset<Ngram> gatherNgram(ESIndex index, String[] ids, NgramTemplate template){
        Multiset<Ngram> multiset = ConcurrentHashMultiset.create();
        String field = template.getField();
        Arrays.stream(ids).parallel().forEach(id -> {
            Map<Integer,String> termVector = index.getTermVectorFromIndex(field, id);
            add(termVector,multiset,template);
        });
        return multiset;
    }

    public static Multiset<Ngram> gatherNgram(ESIndex index, String[] ids, NgramTemplate template, int minDF){
        Multiset<Ngram> multiset = ConcurrentHashMultiset.create();
        String field = template.getField();
        Arrays.stream(ids).parallel().forEach(id -> {
            Map<Integer,String> termVector = index.getTermVectorFromIndex(field, id);
            add(termVector,multiset,template);
        });
        Multiset<Ngram> filtered = ConcurrentHashMultiset.create();
        for (Multiset.Entry entry: multiset.entrySet()){
            Ngram ngram = (Ngram)entry.getElement();
            int count = entry.getCount();
            if (count>=minDF){
                filtered.add(ngram,count);
            }
        }
        return filtered;
    }

    /**
     * gather ngrams with document frequency >= threshold
     * @param index
     * @param ids
     * @param n
     * @param minDf
     * @return
     * @throws Exception
     */
    public static List<String> gatherNgrams(ESIndex index,String field,
                                     String[] ids, int n, int minDf) throws Exception{
        Map<String,Integer> counts = new ConcurrentHashMap<>();
        Arrays.stream(ids).parallel().forEach(id -> {
            Map<Integer,String> termVector = index.getTermVectorFromIndex(field, id);
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



    // each ngram is counted only once in each doc
     private static void add(List<String> source, Multiset<Ngram> multiset, String field, int slop, List<Integer> template){
        Multiset<Ngram> multiSetForDoc = ConcurrentHashMultiset.create();
        for (int i=0;i<source.size();i++){
            if(i+template.get(template.size()-1)<source.size()){
                List<String> list = new ArrayList<>();
                for (int j: template){
                    list.add(source.get(i+j));
                }
                Ngram ngram = new Ngram();
                ngram.setNgram(Ngram.toNgramString(list));
                ngram.setSlop(slop);
                ngram.setField(field);
                ngram.setInOrder(true);
                multiSetForDoc.setCount(ngram,1);
            }
        }
         multiset.addAll(multiSetForDoc);
    }

    public static void add(List<String> source, Multiset<Ngram> multiset, NgramTemplate template){
        for (List<Integer> list: template.getPositionTemplate()){
            add(source,multiset,template.getField(),template.getSlop(),list);
        }
    }

    public static void add(Map<Integer, String> termVector, Multiset<Ngram> multiset, NgramTemplate template){
        Comparator<Map.Entry<Integer,String>> comparator = Comparator.comparing(Map.Entry::getKey);
        List<String> source = termVector.entrySet().stream()
                .sorted(comparator).map(Map.Entry::getValue).collect(Collectors.toList());
        add(source,multiset,template);
    }
}
