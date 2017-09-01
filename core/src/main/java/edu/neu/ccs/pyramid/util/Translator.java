package edu.neu.ccs.pyramid.util;

import java.io.Serializable;
import java.util.Collection;
import java.util.HashMap;
import java.util.Map;

/**
 * Created by chengli on 12/15/15.
 */
public class Translator<T> implements Serializable{
    private static final long serialVersionUID = 1L;
    private Map<T,Integer> objToIndex;
    private Map<Integer,T> indexToObj;
    private int startIndex;
    private int currentIndex;

    public Translator(int startIndex) {
        this.objToIndex = new HashMap<>();
        this.indexToObj = new HashMap<>();
        this.startIndex = startIndex;
        this.currentIndex = startIndex;
    }

    public Translator() {
        this(0);
    }

    public synchronized void add(T obj){
        if (!objToIndex.containsKey(obj)){
            objToIndex.put(obj,currentIndex);
            indexToObj.put(currentIndex,obj);
            currentIndex ++;
        }
    }

    public synchronized void addAll(Collection<T> all){
        all.forEach(this::add);
    }

    public int getIndex(T obj){
        return objToIndex.get(obj);
    }

    public T getObj(int index){
        return indexToObj.get(index);
    }

    public int getStartIndex() {
        return startIndex;
    }

    public int size(){
        return this.indexToObj.size();
    }


    @Override
    public String toString() {
        final StringBuilder sb = new StringBuilder("Translator{");
        sb.append("indexToObj=").append(indexToObj);
        sb.append('}');
        return sb.toString();
    }
}
