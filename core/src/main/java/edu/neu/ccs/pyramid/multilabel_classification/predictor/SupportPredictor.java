package edu.neu.ccs.pyramid.multilabel_classification.predictor;

import edu.neu.ccs.pyramid.calibration.IdentityLabelCalibrator;
import edu.neu.ccs.pyramid.calibration.LabelCalibrator;
import edu.neu.ccs.pyramid.dataset.MultiLabel;
import edu.neu.ccs.pyramid.multilabel_classification.MultiLabelClassifier;
import edu.neu.ccs.pyramid.multilabel_classification.PluginPredictor;
import edu.neu.ccs.pyramid.util.Pair;
import org.apache.mahout.math.Vector;

import java.util.Comparator;
import java.util.List;
import java.util.stream.Collectors;

public class SupportPredictor implements PluginPredictor<MultiLabelClassifier.ClassProbEstimator> {

    private static final long serialVersionUID = 1L;

    MultiLabelClassifier.ClassProbEstimator classifier;
    LabelCalibrator labelCalibrator;
    List<MultiLabel> support;

    public List<MultiLabel> getSupport() {
        return support;
    }

    public SupportPredictor(ClassProbEstimator classifier, LabelCalibrator labelCalibrator, List<MultiLabel> support) {
        this.classifier = classifier;
        this.labelCalibrator = labelCalibrator;
        this.support = support;
    }

    public SupportPredictor(ClassProbEstimator classifier, List<MultiLabel> support) {
        this.classifier = classifier;
        this.support = support;
        this.labelCalibrator = new IdentityLabelCalibrator();
    }

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

    @Override
    public ClassProbEstimator getModel() {
        return classifier;
    }

    @Override
    public MultiLabel predict(Vector vector) {
        double[] uncali = classifier.predictClassProbs(vector);
        double[] cali = labelCalibrator.calibratedClassProbs(uncali);
        return predict(cali,support);
    }
}
