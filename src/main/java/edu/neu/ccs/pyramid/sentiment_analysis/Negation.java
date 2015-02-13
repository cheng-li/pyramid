package edu.neu.ccs.pyramid.sentiment_analysis;

import edu.neu.ccs.pyramid.util.NgramUtil;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

/**
 * Created by chengli on 2/11/15.
 */
public class Negation {
    public static final Set<String> negations = defineNegations();

    public static boolean containsNegation(String ngram){
        for (String term : ngram.split(" ")){
            if (negations.contains(term)){
                return true;
            }
        }
        return false;
    }

    public static boolean startsWithNegation(String ngram){
        if (negations.contains(ngram.split(" ")[0])){
            return true;
        }
        return false;
    }

    public static boolean endsWithNegation(String ngram){
        String[] split = ngram.split(" ");
        if (negations.contains(split[split.length-1])){
            return true;
        }
        return false;
    }



    public static String removeNegation(String ngram){
        List<String> list = new ArrayList<>();
        for (String term: ngram.split(" ")){
            if (!negations.contains(term)){
                list.add(term);
            }
        }
        return NgramUtil.toNgram(list);
    }

    private static Set<String> defineNegations(){
        Set<String> set = new HashSet<>();
        set.add("not");
        set.add("don't");
        set.add("no");
        set.add("isn't");
        set.add("aren't");
        set.add("wasn't");
        set.add("didn't");
        return set;

    }
}
