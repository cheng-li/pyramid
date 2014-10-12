package edu.neu.ccs.pyramid.dataset;

import java.io.Serializable;
import java.util.Arrays;
import java.util.HashSet;
import java.util.Set;
import java.util.stream.Collectors;

/**
 * Created by chengli on 9/27/14.
 */
public class MultiLabel implements Serializable{
    private static final long serialVersionUID = 1L;
    private Set<Integer> labels;
    private boolean[] labelsVector;

    public MultiLabel(int numClasses) {
        this.labels = new HashSet<>();
        this.labelsVector= new boolean[numClasses];
    }

    public MultiLabel addLabel(int k){
        this.labels.add(k);
        this.labelsVector[k]=true;
        return this;
    }

    public boolean matchClass(int k){
        return labelsVector[k];
    }

    public Set<Integer> getMatchedLabels(){
        return labels;
    }

    public int getNumClasses(){
        return this.labelsVector.length;
    }

    @Override
    public String toString() {
        return labels.stream().sorted().collect(Collectors.toList()).toString();
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;

        MultiLabel that = (MultiLabel) o;

        if (!Arrays.equals(labelsVector, that.labelsVector)) return false;

        return true;
    }

    @Override
    public int hashCode() {
        return Arrays.hashCode(labelsVector);
    }
}
