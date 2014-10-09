package edu.neu.ccs.pyramid.classification;

import edu.neu.ccs.pyramid.dataset.FeatureRow;

import java.util.ArrayList;
import java.util.List;

/**
 * Created by chengli on 9/9/14.
 */
public class Voting implements Classifier{
    private int numClasses;
    private List<Classifier> classifiers;

    public Voting(int numClasses) {
        this.numClasses = numClasses;
        this.classifiers = new ArrayList<>();
    }

    public void add(Classifier classifier){
        this.classifiers.add(classifier);
    }

    public int predict(FeatureRow featureRow){
        int[] votes = new int[this.numClasses];
        for (Classifier classifier: this.classifiers){
            int prediction = classifier.predict(featureRow);
            votes[prediction] += 1;
        }
        int max = 0;
        int out = 0;
        for (int i=0;i<this.numClasses;i++){
            if (votes[i]>max){
                max = votes[i];
                out = i;
            }
        }
        return out;
    }

    @Override
    public int getNumClasses() {
        return this.numClasses;
    }
}
