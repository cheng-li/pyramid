package edu.neu.ccs.pyramid.dataset;

import java.util.Arrays;
import java.util.HashSet;
import java.util.Set;

/**
 * Created by chengli on 9/27/14.
 */
public class MultiLabel {
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

    @Override
    public String toString() {
        return
                "labels=" + labels +
                ", labelsVector=" + Arrays.toString(labelsVector)
                ;
    }
}
