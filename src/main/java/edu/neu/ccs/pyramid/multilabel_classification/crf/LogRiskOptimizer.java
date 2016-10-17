package edu.neu.ccs.pyramid.multilabel_classification.crf;

import edu.neu.ccs.pyramid.dataset.MultiLabel;
import edu.neu.ccs.pyramid.dataset.MultiLabelClfDataSet;
import edu.neu.ccs.pyramid.multilabel_classification.MLScorer;
import edu.neu.ccs.pyramid.optimization.GradientDescent;
import edu.neu.ccs.pyramid.optimization.LBFGS;
import edu.neu.ccs.pyramid.optimization.Optimizer;
import edu.neu.ccs.pyramid.optimization.Terminator;
import edu.neu.ccs.pyramid.util.MathUtil;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

import java.util.Arrays;
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
    private Terminator terminator;
    private String optimizer="LBFGS";


    public LogRiskOptimizer(MultiLabelClfDataSet dataSet, MLScorer mlScorer, CMLCRF crf, double variance,
                            boolean expScore, boolean multiplyScore, double scoreMultiplier, double power) {
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

                scores[i][j] = Math.pow(scores[i][j], power);

            }
        }
        this.targets = new double[dataSet.getNumDataPoints()][combinations.size()];
        this.probabilities = new double[dataSet.getNumDataPoints()][combinations.size()];
        this.updateProbabilities();
        this.terminator = new Terminator();
        if (logger.isDebugEnabled()){
            logger.debug("finish constructor");
        }
        // todo
//        System.out.println("scores");
//        System.out.println(Arrays.toString(scores[0]));
        double sum = 0;
        for (int i=0;i<dataSet.getNumDataPoints();i++){
            sum += MathUtil.arraySum(scores[i]);
        }
//        System.out.println("score sum = "+sum);
    }


    public void setOptimizer(String optimizer) {
        this.optimizer = optimizer;
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


    public void optimize(){

        while(!terminator.shouldTerminate()){
            iterate();
        }
    }

    public Terminator getTerminator() {
        return terminator;
    }

    public void iterate(){
        updateTargets();
        updateModel();
        updateProbabilities();
        double objective = objective();
        System.out.println("objective = "+objective);
        terminator.add(objective);
    }

    public void iteratePartial(){
        updateTargets();
        updateModelPartial();
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
        //todo

        Optimizer opt = null;
        switch (optimizer){
            case "LBFGS":
                opt = new LBFGS(klLoss);
                break;
            case "GD":
                opt = new GradientDescent(klLoss);
                break;
            default:
                throw new IllegalArgumentException("unknown");
        }
        opt.optimize();


        if (logger.isDebugEnabled()){
            logger.debug("finish updateModel()");
        }
    }

    private void updateModelPartial(){
        if (logger.isDebugEnabled()){
            logger.debug("start updateModelPartial()");
        }
        KLLoss klLoss = new KLLoss(crf, dataSet, targets, variance);
        //todo

        Optimizer opt = null;
        switch (optimizer){
            case "LBFGS":
                opt = new LBFGS(klLoss);
                break;
            case "GD":
                opt = new GradientDescent(klLoss);
                break;
            default:
                throw new IllegalArgumentException("unknown");
        }
        opt.getTerminator().setMaxIteration(10);
        opt.optimize();


        if (logger.isDebugEnabled()){
            logger.debug("finish updateModelPartial()");
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

    public String objectiveDetail(){
        double obj= IntStream.range(0, dataSet.getNumDataPoints()).parallel()
                .mapToDouble(this::objective).sum();
        double penalty =  penalty();
        StringBuilder sb = new StringBuilder();
        sb.append("empirical loss = "+obj).append("\n");
        sb.append("regularization penalty = "+penalty).append("\n");
        sb.append("total objective = "+(obj+penalty)).append("\n");
        return sb.toString();
    }

    // regularization
    private double penalty(){
        KLLoss klLoss = new KLLoss(crf, dataSet, targets, variance);
        return klLoss.getPenalty();
    }
}
