package edu.neu.ccs.pyramid.eval;

import edu.neu.ccs.pyramid.dataset.MultiLabel;
import edu.neu.ccs.pyramid.dataset.MultiLabelClfDataSet;
import edu.neu.ccs.pyramid.multilabel_classification.MultiLabelClassifier;

import java.util.HashSet;
import java.util.List;
import java.util.Set;
import java.util.stream.IntStream;

/**
 * (A and B)/(A or B)
 * Created by chengli on 10/12/14.
 */
public class Overlap {

    public static double overlap(MultiLabelClassifier classifier, MultiLabelClfDataSet dataSet){
        return overlap(dataSet.getMultiLabels(),classifier.predict(dataSet));
    }

    public static double overlap(MultiLabel[] multiLabels, MultiLabel[] predictions){
        return IntStream.range(0,multiLabels.length).parallel()
                .mapToDouble(i -> overlap(multiLabels[i],predictions[i]))
                .average().getAsDouble();
    }


    private static double overlap(MultiLabel multiLabel1, MultiLabel multiLabel2){
        Set<Integer> set1 = multiLabel1.getMatchedLabels();
        Set<Integer> set2 = multiLabel2.getMatchedLabels();
        Set<Integer> union = new HashSet<>();
        union.addAll(set1);
        union.addAll(set2);
        Set<Integer> itersection = new HashSet<>();
        itersection.addAll(set1);
        itersection.retainAll(set2);
        if (union.size()==0){
            return 1;
        }
        return ((double)itersection.size())/union.size();
    }
}
