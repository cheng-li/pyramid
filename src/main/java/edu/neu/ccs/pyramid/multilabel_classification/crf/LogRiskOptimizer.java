package edu.neu.ccs.pyramid.multilabel_classification.crf;

import edu.neu.ccs.pyramid.dataset.MultiLabel;
import edu.neu.ccs.pyramid.dataset.MultiLabelClfDataSet;
import edu.neu.ccs.pyramid.multilabel_classification.MLScorer;
import edu.neu.ccs.pyramid.optimization.LBFGS;
import edu.neu.ccs.pyramid.util.MathUtil;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

import java.util.List;
import java.util.stream.IntStream;

/**
 * Created by chengli on 9/28/16.
 */
public class LogRiskOptimizer {
    private static final Logger logger = LogManager.getLogger();
    private MultiLabelClfDataSet dataSet;
    private CMLCRF crf;
    // size = [num data][num combination]
    private double[][] targets;
    // todo
    // should be the same as crf combination
    private List<MultiLabel> combinations;
    // size = [num data][num combination]
    private double[][] scores;
    private double variance;
    // size = [num data][num combination]
    private double[][] probabilities;
    private MLScorer mlScorer;
    private boolean expScore = false;
    private boolean multiplyScore = false;
    private double scoreMultiplier = 1;


    public LogRiskOptimizer(MultiLabelClfDataSet dataSet, MLScorer mlScorer, CMLCRF crf, double variance,
                            boolean expScore, boolean multiplyScore, double scoreMultiplier) {
        this.dataSet = dataSet;
        this.variance = variance;
        this.crf = crf;
        this.mlScorer = mlScorer;
        this.combinations = crf.getSupportCombinations();
        this.scores = new double[dataSet.getNumDataPoints()][combinations.size()];
        for (int i=0;i<dataSet.getNumDataPoints();i++){
            for (int j=0;j<combinations.size();j++){
                MultiLabel truth = dataSet.getMultiLabels()[i];
                MultiLabel combination = combinations.get(j);
                double f = mlScorer.score(dataSet.getNumClasses(),truth,combination);
                scores[i][j] = f;
                // todo the order matters here
                if (expScore){
                    scores[i][j] = Math.exp(scores[i][j]);
                }

                if (multiplyScore){
                    scores[i][j] = scores[i][j]*scoreMultiplier;
                }

            }
        }
        this.targets = new double[dataSet.getNumDataPoints()][combinations.size()];
        this.probabilities = new double[dataSet.getNumDataPoints()][combinations.size()];
        this.updateProbabilities();
        if (logger.isDebugEnabled()){
            logger.debug("finish constructor");
        }
    }


    private void updateProbabilities(int dataPointIndex){
        probabilities[dataPointIndex] = crf.predictCombinationProbs(dataSet.getRow(dataPointIndex));
    }

    private void updateProbabilities(){
        if (logger.isDebugEnabled()){
            logger.debug("start updateProbabilities()");
        }
        IntStream.range(0, dataSet.getNumDataPoints()).parallel().forEach(this::updateProbabilities);
        if (logger.isDebugEnabled()){
            logger.debug("finish updateProbabilities()");
        }
    }

    private  void updateTargets(int dataPointIndex){
        double[] probs = probabilities[dataPointIndex];
        double[] product = new double[probs.length];
        double[] s = this.scores[dataPointIndex];
        for (int j=0;j<probs.length;j++){
            product[j] = probs[j]*s[j];
        }

        double denominator = MathUtil.arraySum(product);
        for (int j=0;j<probs.length;j++){
            targets[dataPointIndex][j] = product[j]/denominator;
        }
    }


    public void iterate(){
        updateTargets();
        updateModel();
        updateProbabilities();
    }

    private void updateTargets(){
        if (logger.isDebugEnabled()){
            logger.debug("start updateTargets()");
        }
        IntStream.range(0, dataSet.getNumDataPoints()).parallel().forEach(this::updateTargets);
        if (logger.isDebugEnabled()){
            logger.debug("finish updateTargets()");
        }
    }


    private void updateModel(){
        if (logger.isDebugEnabled()){
            logger.debug("start updateModel()");
        }
        KLLoss klLoss = new KLLoss(crf, dataSet, targets, variance);
        LBFGS lbfgs = new LBFGS(klLoss);
        lbfgs.optimize();
        if (logger.isDebugEnabled()){
            logger.debug("finish updateModel()");
        }
    }


    private double objective(int dataPointIndex){
        double sum = 0;
        double[] p = probabilities[dataPointIndex];
        double[] s = scores[dataPointIndex];
        for (int j=0;j<p.length;j++){
            sum += p[j]*s[j];
        }
        return -Math.log(sum);
    }

    public double objective(){
        if (logger.isDebugEnabled()){
            logger.debug("start objective()");
        }
        double obj= IntStream.range(0, dataSet.getNumDataPoints()).parallel()
                .mapToDouble(this::objective).sum();
        if (logger.isDebugEnabled()){
            logger.debug("finish obj");
        }
        double penalty =  penalty();
        if (logger.isDebugEnabled()){
            logger.debug("finish penalty");
        }
        if (logger.isDebugEnabled()){
            logger.debug("finish objective()");
        }
        return obj+penalty;
    }

    // regularization
    private double penalty(){
        KLLoss klLoss = new KLLoss(crf, dataSet, targets, variance);
        return klLoss.getPenalty();
    }
}
