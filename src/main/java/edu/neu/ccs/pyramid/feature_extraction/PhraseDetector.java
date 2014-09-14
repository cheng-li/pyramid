package edu.neu.ccs.pyramid.feature_extraction;

import java.util.HashSet;
import java.util.Map;
import java.util.Set;

/**
 * Created by chengli on 9/13/14.
 */
public class PhraseDetector {

    public static Set<String> getPhrases(Map<Integer, String> termVector, Set<String> seeds){
        Set<String> phrases = new HashSet<>();
        for (Map.Entry<Integer,String> entry: termVector.entrySet()){
            int pos = entry.getKey();
            String term = entry.getValue();
            if (seeds.contains(term)){
                phrases.addAll(getPhrases(termVector,pos));
            }
        }
        return phrases;
    }

    /**
     * just bigrams for now
     * @param termVector
     * @param pos
     * @return
     */
    public static Set<String> getPhrases(Map<Integer, String> termVector, int pos){
        Set<String> phrases = new HashSet<>();
        String term = termVector.get(pos);
        if (termVector.containsKey(pos-1)){
            phrases.add(termVector.get(pos-1)+" "+term);
        }

        if (termVector.containsKey(pos+1)){
            phrases.add(term +" "+termVector.get(pos+1));
        }

        return phrases;
    }
}
