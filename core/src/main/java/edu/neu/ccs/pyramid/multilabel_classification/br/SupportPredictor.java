package edu.neu.ccs.pyramid.multilabel_classification.br;

import edu.neu.ccs.pyramid.dataset.MultiLabel;
import edu.neu.ccs.pyramid.util.Pair;

import java.util.Comparator;
import java.util.List;
import java.util.stream.Collectors;

public class SupportPredictor {
    public static MultiLabel predict(double[] marginals, List<MultiLabel> support){

        return support.stream().map(m->new Pair<>(m, prob(marginals, m))).max(Comparator.comparing(Pair::getSecond))
                .get().getFirst();
    }


    public static List<Pair<MultiLabel,Double>> topKSetsAndProbs(double[] marginals, List<MultiLabel> support, int top){
        Comparator<Pair<MultiLabel,Double>> comparator = Comparator.comparing(Pair::getSecond);
        return support.stream().map(m->new Pair<>(m, prob(marginals, m))).sorted(comparator.reversed())
                .limit(top).collect(Collectors.toList());
    }

    private static double prob(double[] marginals, MultiLabel multiLabel){
        double p = 1;
        for (int l=0;l<marginals.length;l++){
            if (multiLabel.matchClass(l)){
                p*= marginals[l];
            } else {
                p*= (1-marginals[l]);
            }
        }
        return p;
    }
}
