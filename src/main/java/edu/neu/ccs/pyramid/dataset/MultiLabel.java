package edu.neu.ccs.pyramid.dataset;

import java.util.Arrays;
import java.util.HashSet;
import java.util.Set;

/**
 * Created by chengli on 9/27/14.
 */
public class MultiLabel {
    private Set<Integer> labels;
    private int[] labelsVector;

    public MultiLabel(int numClasses) {
        this.labels = new HashSet<>();
        this.labelsVector= new int[numClasses];
    }

    public MultiLabel addLabel(int k){
        this.labels.add(k);
        this.labelsVector[k]=1;
        return this;
    }

    public int getLabelForClass(int k){
        return labelsVector[k];
    }

    public Set<Integer> getMatchedLabels(){
        return labels;
    }

    @Override
    public String toString() {
        return
                "labels=" + labels +
                ", labelsVector=" + Arrays.toString(labelsVector)
                ;
    }
}
