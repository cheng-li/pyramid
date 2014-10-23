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
    private static final long serialVersionUID = 2L;
    private Set<Integer> labels;


    public MultiLabel() {
        this.labels = new HashSet<>();
    }

    public MultiLabel addLabel(int k){
        this.labels.add(k);
        return this;
    }

    public boolean matchClass(int k){
        return labels.contains(k);
    }

    public Set<Integer> getMatchedLabels(){
        return labels;
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

        if (!labels.equals(that.labels)) return false;

        return true;
    }

    @Override
    public int hashCode() {
        return labels.hashCode();
    }
}
