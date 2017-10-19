package edu.neu.ccs.pyramid.multilabel_classification.imlgb;

import edu.neu.ccs.pyramid.regression.*;
import edu.neu.ccs.pyramid.regression.regression_tree.RegressionTree;
import edu.neu.ccs.pyramid.regression.regression_tree.TreeRule;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.stream.Collectors;

public class LabelModel {
    String labelName;
    double priorSccore;
    List<GeneralTreeRule> rules;


    public String getLabelName() {
        return labelName;
    }

    public double getPriorSccore() {
        return priorSccore;
    }

    public List<GeneralTreeRule> getRules() {
        return rules;
    }

    public LabelModel(IMLGradientBoosting boosting, int classIndex) {
        labelName = boosting.getLabelTranslator().toExtLabel(classIndex);
        List<Regressor> regressors = boosting.getRegressors(classIndex);
        List<GeneralTreeRule> allRules = new ArrayList<>();
        for (Regressor regressor : regressors) {
            if (regressor instanceof ConstantRegressor) {
                this.priorSccore = ((ConstantRegressor)regressor).getScore();
            }

            if (regressor instanceof RegressionTree) {
                RegressionTree tree = (RegressionTree) regressor;
                allRules.addAll(tree.getRules());
            }
        }
        Comparator<GeneralTreeRule> comparator = Comparator.comparing(decision -> Math.abs(decision.getScore()));
        rules = GeneralTreeRule.merge(allRules).stream().sorted(comparator.reversed())
                .collect(Collectors.toList());
    }
}
