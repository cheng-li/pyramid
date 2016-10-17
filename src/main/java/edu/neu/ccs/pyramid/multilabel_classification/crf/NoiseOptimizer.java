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
 * Created by chengli on 10/17/16.
 */
public class NoiseOptimizer {
    private static final Logger logger = LogManager.getLogger();
    private MultiLabelClfDataSet dataSet;
    private CMLCRF crf;
    // size = [num data][num combination]
    private double[][] targets;
    // size = [num data][num combination]
    private double[][] transformProbs;
    // todo
    // should be the same as crf combination
    private List<MultiLabel> combinations;
    private double variance;
    // size = [num data][num combination]
    private double[][] probabilities;
    private Terminator terminator;
    private String optimizer="LBFGS";
    private double[] alphas;


    public NoiseOptimizer(MultiLabelClfDataSet dataSet, CMLCRF crf, double variance) {
        this.dataSet = dataSet;
        this.variance = variance;
        this.crf = crf;
        this.combinations = crf.getSupportCombinations();
        this.targets = new double[dataSet.getNumDataPoints()][combinations.size()];
        this.probabilities = new double[dataSet.getNumDataPoints()][combinations.size()];
        this.transformProbs = new double[dataSet.getNumDataPoints()][combinations.size()];
        this.alphas = new double[dataSet.getNumClasses()];
        Arrays.fill(alphas, 0.9);
        this.updateTransformProbs();
        this.updateProbabilities();
        this.terminator = new Terminator();
        if (logger.isDebugEnabled()){
            logger.debug("finish constructor");
        }
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


    private void updateTransformProbs(){
        IntStream.range(0, dataSet.getNumDataPoints()).parallel()
                .forEach(this::updateTransformProbs);
    }


    private void updateTransformProbs(int dataPoint){
        for (int c=0;c<combinations.size();c++){
            updateTransformProb(dataPoint, c);
        }
    }

    private void updateTransformProb(int dataPoint, int comIndex){
        MultiLabel labels = dataSet.getMultiLabels()[dataPoint];
        MultiLabel candidate = combinations.get(comIndex);
        if (labels.isSubsetOf(candidate)){
            double prod = 1;
            for (int l: candidate.getMatchedLabels()){
                if (labels.matchClass(l)){
                    prod *= alphas[l];
                } else {
                    prod *= (alphas[l]);
                }
            }
            transformProbs[dataPoint][comIndex] = prod;
        } else {
            transformProbs[dataPoint][comIndex] = 0;
        }
    }


    private void updateAlphas(){
        IntStream.range(0, dataSet.getNumClasses()).parallel()
                .forEach(this::updateAlpha);
    }

    private void updateAlpha(int classIndex){
        double numerator = 0;
        double denominator = 0;

        MultiLabel[] multiLabels = dataSet.getMultiLabels();

        for (int c=0;c<combinations.size();c++){
            if (combinations.get(c).matchClass(classIndex)){
                for (int i=0;i<dataSet.getNumDataPoints();i++){
                    MultiLabel multiLabel = multiLabels[i];
                    denominator += targets[i][c];
                    if (multiLabel.matchClass(classIndex)){
                        numerator += targets[i][c];
                    }
                }
            }
        }
        alphas[classIndex] = numerator/denominator;
    }

    private  void updateTargets(int dataPointIndex){
        double[] probs = probabilities[dataPointIndex];
        double[] product = new double[probs.length];
        double[] s = this.transformProbs[dataPointIndex];
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
        updateAlphas();
        updateTransformProbs();
        updateModel();
        updateProbabilities();
        double objective = objective();
        System.out.println("objective = "+objective);
        terminator.add(objective);
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
        double[] s = transformProbs[dataPointIndex];
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
