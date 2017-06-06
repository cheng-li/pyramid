package edu.neu.ccs.pyramid.core.multilabel_classification.adaboostmh;

import edu.neu.ccs.pyramid.core.dataset.ScoreMatrix;
import edu.neu.ccs.pyramid.core.regression.ConstantRegressor;
import edu.neu.ccs.pyramid.core.dataset.WeightMatrix;
import edu.neu.ccs.pyramid.core.dataset.MultiLabelClfDataSet;
import edu.neu.ccs.pyramid.core.multilabel_classification.MLPriorProbClassifier;
import edu.neu.ccs.pyramid.core.regression.Regressor;
import edu.neu.ccs.pyramid.core.regression.regression_tree.RegressionTree;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.apache.mahout.math.Vector;

import java.util.Arrays;
import java.util.Comparator;
import java.util.stream.IntStream;

/**
 * Created by chengli on 3/15/15.
 */
public class AdaBoostMHTrainer {
    private static final Logger logger = LogManager.getLogger();
    private AdaBoostMH boosting;
    private ScoreMatrix scoreMatrix;
    private WeightMatrix weightMatrix;
    private MultiLabelClfDataSet dataSet;
    //speed up access; avoid hashing
    private boolean[][] labels;

    public AdaBoostMHTrainer(MultiLabelClfDataSet dataSet, AdaBoostMH boosting) {
        this.dataSet = dataSet;
        this.boosting = boosting;
        this.boosting.setFeatureList(this.dataSet.getFeatureList());
        this.boosting.setLabelTranslator(this.dataSet.getLabelTranslator());
        if (boosting.getRegressors(0).size()==0){
            this.setPriorProbs(dataSet);
        }
        this.scoreMatrix = new ScoreMatrix(dataSet.getNumDataPoints(),dataSet.getNumClasses());
        this.initStagedClassScoreMatrix();
        this.weightMatrix = new WeightMatrix(dataSet.getNumDataPoints(),dataSet.getNumClasses());
        this.updateDistribution();
        this.labels = new boolean[dataSet.getNumDataPoints()][dataSet.getNumClasses()];
        IntStream.range(0,dataSet.getNumDataPoints()).parallel().forEach(i -> {
            for (int k : dataSet.getMultiLabels()[i].getMatchedLabels()) {
                labels[i][k] = true;
            }
        });
    }


    public void iterate(){
        for (int k=0;k<this.boosting.getNumClasses();k++){
            /**
             * parallel by feature
             */
            Regressor regressor = this.fitClassK(k);
            this.boosting.addRegressor(regressor, k);
            /**
             * parallel by data
             */
            this.updateStagedClassScores(regressor,k);
        }
        this.updateDistribution();
    }


    private void setPriorProbs(double[] probs){
        if (probs.length!=this.boosting.getNumClasses()){
            throw new IllegalArgumentException("probs.length!=this.numClasses");
        }
        double average = Arrays.stream(probs).map(Math::log).average().getAsDouble();
        for (int k=0;k<this.boosting.getNumClasses();k++){
            double score = Math.log(probs[k] - average);
            Regressor constant = new ConstantRegressor(score);
            this.boosting.addRegressor(constant, k);
        }
    }

    /**
     * not sure whether this is good for performance
     * start with prior probabilities
     * should be called before setTrainConfig
     */
    private void setPriorProbs(MultiLabelClfDataSet dataSet){
        MLPriorProbClassifier priorProbClassifier = new MLPriorProbClassifier(dataSet.getNumClasses());
        priorProbClassifier.fit(dataSet);
        double[] probs = priorProbClassifier.getClassProbs();
        this.setPriorProbs(probs);
    }

    private void updateDistribution(){
        int numClasses = boosting.getNumClasses();
        IntStream.range(0,dataSet.getNumDataPoints()).parallel().forEach(i->{
            double[] y = new double[numClasses];
            Arrays.fill(y,-1);
            for (int k: dataSet.getMultiLabels()[i].getMatchedLabels()){
                y[k] = 1;
            }
            float[] scores = scoreMatrix.getScoresForData(i);
            for (int k=0;k<numClasses;k++){
                double prob = Math.exp(-1*y[k]*scores[k]);
                weightMatrix.setProbability(i, k, prob);
            }
        });
        weightMatrix.normalize();
    }

    private void initStagedClassScoreMatrix(){
        int numClasses = boosting.getNumClasses();
        for (int k=0;k<numClasses;k++){
            for (Regressor regressor: boosting.getRegressors(k)){
                this.updateStagedClassScores(regressor, k);
            }
        }
    }

    private void updateStagedClassScores(Regressor regressor, int k){
        int numDataPoints = dataSet.getNumDataPoints();
        IntStream.range(0, numDataPoints).parallel()
                .forEach(dataIndex -> this.updateStagedClassScore(regressor, k, dataIndex));
    }

    private void updateStagedClassScore(Regressor regressor, int k,
                                        int dataIndex){
        Vector vector = dataSet.getRow(dataIndex);
        double prediction = regressor.predict(vector);
        this.scoreMatrix.increment(dataIndex,k,prediction);
    }

    private RegressionTree fitClassK(int k){
        double[] probs = weightMatrix.getProbsForClass(k);

        double match = IntStream.range(0,dataSet.getNumDataPoints())
                .parallel().filter(i-> labels[i][k])
                .mapToDouble(i -> weightMatrix.getProbsForData(i)[k])
                .sum();

        double notMatch = IntStream.range(0,dataSet.getNumDataPoints())
                .parallel().filter(i -> !labels[i][k])
                .mapToDouble(i -> weightMatrix.getProbsForData(i)[k])
                .sum();

        StumpInfo optimal = IntStream.range(0, dataSet.getNumFeatures())
                .parallel().mapToObj(j -> {
            double matchOccur = 0;
            double notMatchOccur = 0;
            Vector vector = dataSet.getColumn(j);
            for (Vector.Element element : vector.nonZeroes()) {
                int i = element.index();
                double prob = probs[i];
                if (labels[i][k]) {
                    matchOccur += prob;
                } else {
                    notMatchOccur += prob;
                }
            }
            double matchNotOccur = match - matchOccur;
            double notMatchNotOccur = notMatch - notMatchOccur;
            StumpInfo stumpInfo = new StumpInfo();
            stumpInfo.featureIndex = j;
            stumpInfo.matchOccur = matchOccur;
            stumpInfo.matchNotOccur = matchNotOccur;
            stumpInfo.notMatchOccur = notMatchOccur;
            stumpInfo.notMatchNotOccur = notMatchNotOccur;
            return stumpInfo;
        }).min(Comparator.comparing(StumpInfo::getObjective)).get();
        double smooth = 1.0/(dataSet.getNumDataPoints()*dataSet.getNumClasses());
        double leftOutput = 0.5 * Math.log((optimal.matchNotOccur+smooth)
                /(optimal.notMatchNotOccur+smooth));
        double rightOutput = 0.5 * Math.log((optimal.matchOccur+smooth)
                /(optimal.notMatchOccur+smooth));

        RegressionTree tree = RegressionTree.newStump(optimal.featureIndex,0,
                leftOutput,rightOutput);
        tree.setFeatureList(dataSet.getFeatureList());
        return tree;
    }

    private static class StumpInfo{
        private int featureIndex;
        private double matchOccur;
        private double matchNotOccur;
        private double notMatchOccur;
        private double notMatchNotOccur;

        double getObjective(){
            return Math.sqrt(matchOccur*notMatchOccur)
                    + Math.sqrt(matchNotOccur * notMatchNotOccur);
        }

    }
}
