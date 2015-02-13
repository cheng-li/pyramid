package edu.neu.ccs.pyramid.util;

import java.util.List;

/**
 * Created by chengli on 2/11/15.
 */
public class NgramUtil {
    public static String toNgram(List<String> termList){
        String ngram = "";
        for (int i=0;i<termList.size();i++){
            ngram = ngram.concat(termList.get(i));
            if (i<termList.size()-1){
                ngram = ngram.concat(" ");
            }
        }
        return ngram;
    }
}
