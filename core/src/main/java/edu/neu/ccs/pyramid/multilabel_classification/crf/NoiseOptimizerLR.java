package edu.neu.ccs.pyramid.multilabel_classification.crf;

import edu.neu.ccs.pyramid.classification.logistic_regression.LogisticRegression;
import edu.neu.ccs.pyramid.classification.logistic_regression.RidgeLogisticOptimizer;
import edu.neu.ccs.pyramid.dataset.ClfDataSet;
import edu.neu.ccs.pyramid.dataset.ClfDataSetBuilder;
import edu.neu.ccs.pyramid.dataset.MultiLabel;
import edu.neu.ccs.pyramid.dataset.MultiLabelClfDataSet;
import edu.neu.ccs.pyramid.multilabel_classification.Enumerator;
import edu.neu.ccs.pyramid.optimization.GradientDescent;
import edu.neu.ccs.pyramid.optimization.LBFGS;
import edu.neu.ccs.pyramid.optimization.Optimizer;
import edu.neu.ccs.pyramid.optimization.Terminator;
import edu.neu.ccs.pyramid.util.ArgSort;
import edu.neu.ccs.pyramid.util.MathUtil;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.Vector;

import java.util.ArrayList;
import java.util.List;
import java.util.stream.IntStream;

/**
 * Created by yuyuxu on 12/06/16.
 */
public class NoiseOptimizerLR {
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
    // stores bunch of logistic regression
    public List<LogisticRegression> lrTransforms;
    // stores data set for logistic regression for each class
    private List<ClfDataSet> lrDataSet;
    // stores target distribution fo logistic regression for each class
    private double[][][] lrTargets;


    public NoiseOptimizerLR(MultiLabelClfDataSet dataSet, CMLCRF crf, double variance) {
        this.dataSet = dataSet;
        this.variance = variance;
        this.crf = crf;
//        this.combinations = crf.getSupportCombinations();
        this.combinations = Enumerator.enumerate(dataSet.getNumClasses());

//        System.out.println("enumerations:");
//        for(int i=0;i<combinations.size();i++){
//            System.out.println(combinations.get(i).toString());
//        }

        this.targets = new double[dataSet.getNumDataPoints()][combinations.size()];
        this.probabilities = new double[dataSet.getNumDataPoints()][combinations.size()];
        this.transformProbs = new double[dataSet.getNumDataPoints()][combinations.size()];

        // Initialize logistic regressions related variables
        this.lrTransforms = new ArrayList<>();
        this.lrDataSet = new ArrayList<>();
        int numCombination = (int)Math.pow(2,dataSet.getNumClasses());
//        System.out.println("num classes="+dataSet.getNumClasses()+",number combination=" + numCombination+",combinations.size()="+combinations.size());
        if (numCombination != this.combinations.size()) {
            throw new IllegalArgumentException("number of combination should equal!");
        }

//        // Following code just set the true weights to initialize the logistic regression, to see if weights diffuse after few iterations
//        Vector[] weights = new Vector[dataSet.getNumClasses()];
//        for (int k=0;k<dataSet.getNumClasses();k++){
//            Vector vector = new DenseVector(dataSet.getNumClasses());
//            weights[k] = vector;
//        }
//        double w_pos = 10;
//        double w_neg = -1;
//        weights[0].set(0,w_pos);
//        weights[0].set(1,w_neg);
//        weights[0].set(2,w_neg);
//        weights[0].set(3,w_neg);
//
//        weights[1].set(0,w_neg);
//        weights[1].set(1,w_pos);
//        weights[1].set(2,w_neg);
//        weights[1].set(3,w_neg);
//
//        weights[2].set(0,w_neg);
//        weights[2].set(1,w_neg);
//        weights[2].set(2,w_pos);
//        weights[2].set(3,w_neg);
//
//        weights[3].set(0,w_neg);
//        weights[3].set(1,w_neg);
//        weights[3].set(2,w_neg);
//        weights[3].set(3,w_pos);

        this.lrTargets = new double[dataSet.getNumClasses()][dataSet.getNumDataPoints() * numCombination][2];
        for (int i=0; i<dataSet.getNumClasses(); i++) {
            LogisticRegression lr = new LogisticRegression(2, dataSet.getNumClasses(), true);
//            LogisticRegression lr = new LogisticRegression(2, dataSet.getNumClasses());
//            for (int j = 0; j < dataSet.getNumClasses(); j++) {
//                lr.getWeights().getWeightsWithoutBiasForClass(1).set(j, weights[i].get(j));
//            }
//            System.out.println("init lr "+i+":" + lr);
            this.lrTransforms.add(lr);
            this.lrDataSet.add(this.buildLrData(i));
            this.lrTargets[i] = this.buildLrTargets(i);
        }

        this.updateTransformProbs();
        this.updateProbabilities();
        this.terminator = new Terminator();

        if (logger.isDebugEnabled()){
            logger.debug("finish constructor");
        }
    }

    private double[][] buildLrTargets(int classIndex) {
        int numCombination = this.combinations.size();
        double[][] targets = new double[this.dataSet.getNumDataPoints() * numCombination][2];
        for (int i=0;i<dataSet.getNumDataPoints();i++) {
            int labelToSet = 0;
            if (dataSet.getMultiLabels()[i].matchClass(classIndex)) {
                labelToSet = 1;
            }
            for (int j=0;j<numCombination;j++) {
                if (labelToSet == 1) {
                    targets[i * numCombination + j][0] = 0;
                    targets[i * numCombination + j][1] = 1;
                } else {
                    targets[i * numCombination + j][0] = 1;
                    targets[i * numCombination + j][1] = 0;
                }
            }
        }

//        System.out.println("buildLrTargets,index="+classIndex);
//        for(double[] row : targets) {
//            for (double i : row) {
//                System.out.print(i);
//                System.out.print("\t");
//            }
//            System.out.println();
//        }

        return targets;
    }

    private ClfDataSet buildLrData(int classIndex) {
        // Generate data set for current class
        int numCombination = this.combinations.size();
        ClfDataSet lrDataSet = ClfDataSetBuilder.getBuilder()
                .numDataPoints(dataSet.getNumDataPoints() * numCombination)
                .numFeatures(dataSet.getNumClasses())
                .numClasses(2)
                .dense(true)
                .missingValue(false)
                .build();

        for (int i=0;i<this.dataSet.getNumDataPoints();i++){
            int labelToSet = 0;
            if (this.dataSet.getMultiLabels()[i].matchClass(classIndex)) {
                labelToSet = 1;
            }
            for (int k = 0; k < numCombination; k++) {
                // set feature
                for (int j=0; j<this.dataSet.getNumClasses(); j++) {
                    if (this.combinations.get(k).matchClass(j)) {
//                        lrDataSet.setFeatureValue(i * numCombination + k, j, 1);
                        lrDataSet.setFeatureValue(i * numCombination + k, j, 0.5);
                    } else {
                        lrDataSet.setFeatureValue(i * numCombination + k, j, -0.5);
                    }
                }
                // set label
                lrDataSet.setLabel(i * numCombination + k, labelToSet);
            }
        }

//        System.out.println("buildLrData,index="+classIndex);
//        System.out.println(lrDataSet);

        return lrDataSet;
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
        Vector toMinus = new DenseVector(dataSet.getNumClasses());
        for (int i=0;i<dataSet.getNumClasses();i++){
            toMinus.set(i,0.5);
        }
        double prod = 1;
        for (int l = 0; l < dataSet.getNumClasses(); l++) {
            if (labels.matchClass(l)) {
                prod *= this.lrTransforms.get(l).predictClassProb(candidate.toVector(dataSet.getNumClasses()).minus(toMinus), 1);
            } else {
                prod *= this.lrTransforms.get(l).predictClassProb(candidate.toVector(dataSet.getNumClasses()).minus(toMinus), 0);
            }
        }
        transformProbs[dataPoint][comIndex] = prod;
    }

    private void updateAlphas(){
        IntStream.range(0, dataSet.getNumClasses()).parallel()
                .forEach(this::updateAlpha);
    }

    private void updateAlpha(int classIndex){
        int numCombination = this.combinations.size();

        // Generate weights for each expanded data point
        double[] weights = new double[dataSet.getNumDataPoints() * numCombination];
        for (int i=0;i<dataSet.getNumDataPoints();i++) {
            for (int j=0;j<numCombination;j++) {
                weights[i*numCombination + j] = targets[i][j];
            }
        }

        // Train weighted logistic regression with l2
        RidgeLogisticOptimizer optimizer = new RidgeLogisticOptimizer(this.lrTransforms.get(classIndex),this.lrDataSet.get(classIndex),weights,this.lrTargets[classIndex],1000.0,true);
        optimizer.getOptimizer().getTerminator().setMaxIteration(10000).setMode(Terminator.Mode.STANDARD);
//        System.out.println("after initialization");
//        System.out.println("train acc = " + Accuracy.accuracy(this.lrTransforms.get(classIndex), this.lrDataSet.get(classIndex)));
        optimizer.optimize();
//        System.out.println("after training");
//        System.out.println("train acc = " + Accuracy.accuracy(this.lrTransforms.get(classIndex), this.lrDataSet.get(classIndex)));
//        System.out.println("classIndex: " + classIndex + ", weights:" + this.lrTransforms.get(classIndex));
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


    private void printProbWithThreshold(double[] probs, double thresh) {
        int[] indices = ArgSort.argSortDescending(probs);
        for (int i = 0; i < indices.length; i++) {
            if (probs[indices[i]] >= thresh) {
                System.out.println(indices[i]+":"+combinations.get(indices[i]).toString()+":"+probs[indices[i]]);
            }
        }
    }

    public void printInfo() {
        for (int i = 0; i < dataSet.getNumDataPoints(); i++) {
            System.out.println("index=" + i + ",label=" + dataSet.getMultiLabels()[i].toString());
            System.out.println("printing targets ..");
            printProbWithThreshold(targets[i], 0.1);
            System.out.println("printing transformProbs ..");
            printProbWithThreshold(transformProbs[i], 0.1);
            System.out.println("printing probability ..");
            printProbWithThreshold(probabilities[i], 0.1);
        }
    }

    public void iterate(){
        updateTargets();
        System.out.println("finish updateTargets ");
        System.out.println("objective = "+objective());
        updateAlphas();
        System.out.println("finish updateAlphas ");
        System.out.println("objective = "+objective());
        updateTransformProbs();
        System.out.println("finish updateTransformProbs ");
        System.out.println("objective = "+objective());
        updateModel();
        System.out.println("finish updateModel ");
        System.out.println("objective = "+objective());
        updateProbabilities();
        System.out.println("finish updateProbabilities ");
        double objective = objective();
        System.out.println("objective = "+objective);
        terminator.add(objective);
    }


    public void iteratePartial(int modelIterations){
        updateTargets();
        System.out.println("finish updateTargets ");
        System.out.println("objective = "+objective());
        updateAlphas();
        System.out.println("finish updateAlphas ");
        System.out.println("objective = "+objective());
        updateTransformProbs();
        System.out.println("finish updateTransformProbs ");
        System.out.println("objective = "+objective());
        updateModelPartial(modelIterations);
        System.out.println("finish updateModel ");
        System.out.println("objective = "+objective());
        updateProbabilities();
        System.out.println("finish updateProbabilities ");
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



    private void updateModelPartial(int modelIterations){
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
        opt.getTerminator().setMaxIteration(modelIterations);
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
