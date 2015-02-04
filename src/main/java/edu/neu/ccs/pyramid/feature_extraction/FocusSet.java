package edu.neu.ccs.pyramid.feature_extraction;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

/**
 * Created by chengli on 9/6/14.
 */
public class FocusSet {
    /**
     * dataPointIndices.get(k) = the focus docs for class k
     */
    private List<List<Integer>> dataPointIndices;
    private int numClasses;
    //for multi-label datasets, need to remove duplicates
    private Set<Integer> all;

    public FocusSet(int numClasses) {
        this.numClasses = numClasses;
        this.dataPointIndices = new ArrayList<List<Integer>>(numClasses);
        for (int k = 0;k<numClasses;k++){
            this.dataPointIndices.add(new ArrayList<Integer>());
        }
        this.all = new HashSet<>();
    }

    public void add(int dataPointIndex, int trueLabel){
        this.dataPointIndices.get(trueLabel).add(dataPointIndex);
        this.all.add(dataPointIndex);
    }

    public List<Integer> getDataClassK(int k){
        return this.dataPointIndices.get(k);
    }

    //todo fix format
    public List<Integer> getAll() {
        return new ArrayList<>(all);
    }

    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder();
        sb.append("focus set: \n");
        for (int k=0;k<this.numClasses;k++){
            sb.append("class ").append(k).append(": ");
            sb.append(this.dataPointIndices.get(k).toString()).append("  ");
        }
        return sb.toString();
    }
}
