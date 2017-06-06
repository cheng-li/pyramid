package edu.neu.ccs.pyramid.core.util;

import java.util.List;
import java.util.regex.Pattern;

/**
 * Created by chengli on 5/9/16.
 */
public class ListUtil {

    /**
     * string without [ or ]
     * @param list
     * @return
     */
    public static String toSimpleString(List list){
        return list.toString().replaceAll(Pattern.quote("["),"").replaceAll(Pattern.quote("]"),"");
    }
}
