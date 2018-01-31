package edu.neu.ccs.pyramid.multilabel_classification.crf;

import edu.neu.ccs.pyramid.dataset.MultiLabel;
import edu.neu.ccs.pyramid.dataset.MultiLabelClfDataSet;
import edu.neu.ccs.pyramid.multilabel_classification.PluginPredictor;
import edu.neu.ccs.pyramid.multilabel_classification.plugin_rule.GeneralF1Predictor;
import org.apache.mahout.math.Vector;

import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.DoubleStream;

/**
 * Created by Rainicy on 5/7/16.
 */
public class InstanceF1Predictor implements PluginPredictor<CMLCRF> {
    private CMLCRF cmlcrf;
    private int numClasses;
    private PMatrixIsotonicScaling pMatrixIsotonicScaling;


    private String predictMode;

    public InstanceF1Predictor(CMLCRF model) {
        this.cmlcrf = model;
        this.numClasses = model.getNumClasses();
    }

    public InstanceF1Predictor(CMLCRF model, MultiLabelClfDataSet dataSet, boolean isPair) {
        this.cmlcrf = model;
        this.numClasses = model.getNumClasses();
        if (isPair) {
            this.pMatrixIsotonicScaling = new PMatrixIsotonicScaling(cmlcrf, dataSet);
        }
    }

    @Override
    public MultiLabel predict(Vector vector) {
        if (predictMode.equals("defualt")) {
            List<MultiLabel> supports = cmlcrf.getSupportCombinations();
            double[] probs = cmlcrf.predictCombinationProbs(vector);
            GeneralF1Predictor generalF1Predictor = new GeneralF1Predictor();
            return generalF1Predictor.predict(numClasses, supports, probs);
        }
        return predictByPair(vector);
    }

    private MultiLabel predictByPair(Vector vector) {
        if (this.pMatrixIsotonicScaling == null) {
            throw new RuntimeException("missing istonic scaling.");
        }

        double[] probs = cmlcrf.predictAssignmentProbs(vector, cmlcrf.getSupportCombinations());
        GeneralF1Predictor generalF1Predictor = new GeneralF1Predictor();
        double[][] p = generalF1Predictor.getPMatrix(numClasses, cmlcrf.getSupportCombinations(),
                DoubleStream.of(probs).boxed().collect(Collectors.toList()));
        for (int i=0; i<p.length; i++) {
            for (int j=0; j<p[i].length; j++) {
                p[i][j] = pMatrixIsotonicScaling.calibratedProb(p[i][j]);
            }
        }

        double zeroProb = 0;
        for (int i=0;i< cmlcrf.getSupportCombinations().size();i++){
            if ( cmlcrf.getSupportCombinations().get(i).getMatchedLabels().size()==0){
                zeroProb = probs[i];
                break;
            }
        }
        return generalF1Predictor.predictWithPMatrix(p, zeroProb);

    }

    public GeneralF1Predictor.Analysis showPredictBySupport(Vector vector, MultiLabel truth){
//        System.out.println("support procedure");
        List<MultiLabel> support = cmlcrf.getSupportCombinations();
        double[] probs = cmlcrf.predictCombinationProbs(vector);
        GeneralF1Predictor generalF1Predictor = new GeneralF1Predictor();
        MultiLabel prediction =  generalF1Predictor.predict(cmlcrf.getNumClasses(),support,probs);
        GeneralF1Predictor.Analysis analysis = GeneralF1Predictor.showSupportPrediction(support,probs, truth, prediction, cmlcrf.getNumClasses());
        return analysis;
    }

    @Override
    public CMLCRF getModel() {
        return cmlcrf;
    }


    public String getPredictMode() {
        return predictMode;
    }

    public void setPredictMode(String predictMode) {
        this.predictMode = predictMode;
    }
}
