package edu.neu.ccs.pyramid.util;

import java.util.HashSet;
import java.util.Set;

/**
 * Created by chengli on 12/18/14.
 */
public class SetUtil {

    public static <E> Set<E> union(Set<E> set1, Set<E> set2) {
        Set<E> union = new HashSet<>();
        union.addAll(set1);
        union.addAll(set2);
        return union;
    }

    public static <E> Set<E> intersect(Set<E> set1, Set<E> set2) {
        Set<E> result = new HashSet<>();
        result.addAll(set1);
        result.retainAll(set2);
        return result;
    }

}
