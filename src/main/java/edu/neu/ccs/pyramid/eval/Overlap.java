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

    public static double overlap(double tp, double fp, double fn){
        return SafeDivide.divide(tp,tp+fp+fn,1);
    }

    @Deprecated
    public static double overlap(MultiLabelClassifier classifier, MultiLabelClfDataSet dataSet){
        return overlap(dataSet.getMultiLabels(),classifier.predict(dataSet));
    }

    @Deprecated
    public static double overlap(MultiLabel[] multiLabels, MultiLabel[] predictions){
        return IntStream.range(0,multiLabels.length).parallel()
                .mapToDouble(i -> overlap(multiLabels[i],predictions[i]))
                .average().getAsDouble();
    }

    public static double overlap(MultiLabel multiLabel1, MultiLabel multiLabel2){
        Set<Integer> set1 = multiLabel1.getMatchedLabels();
        Set<Integer> set2 = multiLabel2.getMatchedLabels();
        Set<Integer> union = new HashSet<>();
        union.addAll(set1);
        union.addAll(set2);
        Set<Integer> intersection = new HashSet<>();
        intersection.addAll(set1);
        intersection.retainAll(set2);
        return SafeDivide.divide(intersection.size(),union.size(),1);
    }
}
