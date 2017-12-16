package edu.neu.ccs.pyramid.util;

import java.io.Serializable;

/**
 * Created by chengli on 8/19/14.
 */
public class Pair<A,B> implements Serializable{
    private static final long serialVersionUID = 1L;
    private A first;
    private B second;

    public Pair(A first, B second) {
        this.first = first;
        this.second = second;
    }

    public Pair() {
    }

    public A getFirst() {
        return first;
    }

    public Pair<A, B> setFirst(A first) {
        this.first = first;
        return this;
    }

    public B getSecond() {
        return second;
    }

    public Pair<A, B> setSecond(B second) {
        this.second = second;
        return this;
    }

    @Override
    public String toString() {
        return "("+first+", "+second+")";
    }
}
