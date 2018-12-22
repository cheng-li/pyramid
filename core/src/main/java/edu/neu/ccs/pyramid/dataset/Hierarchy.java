package edu.neu.ccs.pyramid.dataset;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.List;

public class Hierarchy implements Serializable {
    private static final long serialVersionUID = 1L;

    private int numLabels;

    public Hierarchy(int numLabels) {
        this.numLabels = numLabels;
        this.parents = new ArrayList<>(numLabels);
        for (int l=0;l<numLabels;l++){
            parents.add(new ArrayList<>());
        }
    }

    private List<List<Integer>> parents;

    public List<Integer> getParentsForLabel(int label){
        return parents.get(label);
    }

    public void setParentsForLabel(int l, List<Integer> parents){
        this.parents.get(l).clear();
        this.parents.get(l).addAll(parents);
    }


    public boolean satisfy(MultiLabel multiLabel){
        for (int l: multiLabel.getMatchedLabels()){
            List<Integer> parents = getParentsForLabel(l);
            for (int p: parents){
                if (!multiLabel.matchClass(p)){
                    return false;
                }
            }
        }
        return true;
    }


    @Override
    public String toString() {
        final StringBuilder sb = new StringBuilder("Hierarchy\n");
        for (int l=0;l<numLabels;l++){
            sb.append(l).append(": ").append(getParentsForLabel(l)).append("\n");
        }
        return sb.toString();
    }
}
