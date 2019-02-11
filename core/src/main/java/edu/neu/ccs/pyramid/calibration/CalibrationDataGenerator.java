package edu.neu.ccs.pyramid.calibration;

import edu.neu.ccs.pyramid.dataset.MultiLabel;
import edu.neu.ccs.pyramid.dataset.MultiLabelClfDataSet;
import edu.neu.ccs.pyramid.dataset.RegDataSet;
import edu.neu.ccs.pyramid.dataset.RegDataSetBuilder;
import edu.neu.ccs.pyramid.feature.FeatureList;
import edu.neu.ccs.pyramid.multilabel_classification.DynamicProgramming;

import edu.neu.ccs.pyramid.multilabel_classification.MultiLabelClassifier;
import edu.neu.ccs.pyramid.util.Vectors;
import org.apache.mahout.math.Vector;

import java.util.ArrayList;
import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

public class CalibrationDataGenerator {
    private LabelCalibrator labelCalibrator;
    private PredictionFeatureExtractor predictionFeatureExtractor;

    public CalibrationDataGenerator(LabelCalibrator labelCalibrator, PredictionFeatureExtractor predictionFeatureExtractor) {
        this.labelCalibrator = labelCalibrator;
        this.predictionFeatureExtractor = predictionFeatureExtractor;
    }

    public TrainData createCaliTrainingData(MultiLabelClfDataSet calDataSet, MultiLabelClassifier.ClassProbEstimator classProbEstimator, int numCandidates){
        List<CalibrationInstance> instances = IntStream.range(0, calDataSet.getNumDataPoints()).parallel()
                .boxed().flatMap(i -> expand(calDataSet.getRow(i),calDataSet.getMultiLabels()[i], classProbEstimator , i, numCandidates).stream())
                .collect(Collectors.toList());
        return createData(instances);
    }


    public TrainData createCaliTrainingData(MultiLabelClfDataSet calDataSet, List<Vector> uncalibratedLabelScores, int numCandidates){
        List<CalibrationInstance> instances = IntStream.range(0, calDataSet.getNumDataPoints()).parallel()
                .boxed().flatMap(i -> expand(calDataSet.getRow(i),calDataSet.getMultiLabels()[i], Vectors.toArray(uncalibratedLabelScores.get(i)) , i, numCandidates).stream())
                .collect(Collectors.toList());
        return createData(instances);
    }

    private TrainData createData(List<CalibrationInstance> calibrationInstances){
        RegDataSet regDataSet = RegDataSetBuilder.getBuilder()
                .numDataPoints(calibrationInstances.size())
                .numFeatures(calibrationInstances.get(0).vector.size())
                .dense(false)
                .build();
        double[] weights = new double[calibrationInstances.size()];
        for (int i = 0; i< calibrationInstances.size(); i++){
            //todo sparse
            for (int j=0;j<regDataSet.getNumFeatures();j++){
                regDataSet.setFeatureValue(i,j, calibrationInstances.get(i).vector.get(j));
            }
            regDataSet.setLabel(i, calibrationInstances.get(i).correctness);
            weights[i] = calibrationInstances.get(i).weight;
        }

        FeatureList featureList = new FeatureList();
        featureList.addAll(predictionFeatureExtractor.getNames());
        regDataSet.setFeatureList(featureList);

        int numQueries = calibrationInstances.stream().mapToInt(calibrationInstance -> calibrationInstance.queryIndex).max().getAsInt()+1;
        List<List<Integer>> instancesForEachQuery = new ArrayList<>();
        for (int q=0;q<numQueries;q++){
            instancesForEachQuery.add(new ArrayList<>());
        }

        for (int i = 0; i< calibrationInstances.size(); i++){
            int q = calibrationInstances.get(i).queryIndex;
            instancesForEachQuery.get(q).add(i);
        }
        
        return new TrainData(regDataSet, weights, instancesForEachQuery);
    }

    private List<CalibrationInstance> expand(Vector x, MultiLabel groundTruth,
                                             MultiLabelClassifier.ClassProbEstimator classProbEstimator, int queryId, int numCandidates){
        double[] uncalibratedLabelScores = classProbEstimator.predictClassProbs(x);
        return expand(x, groundTruth, uncalibratedLabelScores, queryId, numCandidates);
    }

    private List<CalibrationInstance> expand(Vector x, MultiLabel groundTruth,
                                             double[] uncalibratedLabelScores, int queryId, int numCandidates){
        double[] marginals = labelCalibrator.calibratedClassProbs(uncalibratedLabelScores);
        List<CalibrationInstance> calibrationInstances = new ArrayList<>();

        DynamicProgramming dynamicProgramming = new DynamicProgramming(marginals);

        for (int i=0;i<numCandidates;i++){
            MultiLabel multiLabel = dynamicProgramming.nextHighestVector();
            PredictionCandidate predictionCandidate = new PredictionCandidate();
            predictionCandidate.multiLabel = multiLabel;
            predictionCandidate.labelProbs = marginals;
            predictionCandidate.x = x;
            CalibrationInstance calibrationInstance = createInstance(groundTruth,predictionCandidate);
            calibrationInstance.weight=1;
            calibrationInstance.queryIndex = queryId;
            calibrationInstances.add(calibrationInstance);
        }

        return calibrationInstances;
    }


    public CalibrationInstance createInstance(MultiLabel groundtruth, PredictionCandidate predictionCandidate){
        CalibrationInstance calibrationInstance = new CalibrationInstance();
        calibrationInstance.vector= predictionFeatureExtractor.extractFeatures(predictionCandidate);
        calibrationInstance.correctness = 0;
        if (groundtruth.equals(predictionCandidate.multiLabel)){
            calibrationInstance.correctness=1;
        }
        return calibrationInstance;
    }


    public CalibrationInstance createInstance(MultiLabelClassifier.ClassProbEstimator classProbEstimator, Vector x,
                                              MultiLabel prediction, MultiLabel groundtruth){
        double[] uncalibratedLabelScores = classProbEstimator.predictClassProbs(x);
        return createInstance(x, uncalibratedLabelScores, prediction, groundtruth);
    }

    public CalibrationInstance createInstance(Vector x, double[] uncalibratedLabelScores,
                                              MultiLabel prediction, MultiLabel groundtruth){
        PredictionCandidate predictionCandidate = new PredictionCandidate();
        predictionCandidate.x = x;
        predictionCandidate.multiLabel = prediction;
        predictionCandidate.labelProbs = labelCalibrator.calibratedClassProbs(uncalibratedLabelScores);
        return createInstance(groundtruth,predictionCandidate);
    }

    public static class TrainData{
        public RegDataSet regDataSet;
        public double[] instanceWeights;
        public List<List<Integer>> instancesForEachQuery;


        public TrainData(RegDataSet regDataSet, double[] instanceWeights, List<List<Integer>> instancesForEachQuery) {
            this.regDataSet = regDataSet;
            this.instanceWeights = instanceWeights;
            this.instancesForEachQuery = instancesForEachQuery;
        }
    }

    public static class CalibrationInstance {
        public Vector vector;
        public double correctness;
        public double weight=1;
        public int queryIndex;
    }

}
