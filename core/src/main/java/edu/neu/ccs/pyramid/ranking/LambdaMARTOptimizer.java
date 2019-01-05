package edu.neu.ccs.pyramid.ranking;

import edu.neu.ccs.pyramid.dataset.DataSet;
import edu.neu.ccs.pyramid.optimization.gradient_boosting.GBOptimizer;
import edu.neu.ccs.pyramid.optimization.gradient_boosting.GradientBoosting;
import edu.neu.ccs.pyramid.regression.RegressorFactory;

import java.util.List;
import java.util.stream.IntStream;

public class LambdaMARTOptimizer extends GBOptimizer {
    private double[] relevanceGrades;

    // format instanceIdsInEachQuery.get(query id).get(local instance id in query) = global instance id in dataset
    private List<List<Integer>> instanceIdsInEachQuery;
    private int numQueries;


    public LambdaMARTOptimizer(GradientBoosting boosting, DataSet dataSet, RegressorFactory factory, double[] relevanceGrades, List<List<Integer>> instanceIdsInEachQuery) {
        super(boosting, dataSet, factory);
        this.relevanceGrades = relevanceGrades;
        this.instanceIdsInEachQuery = instanceIdsInEachQuery;
        this.numQueries = instanceIdsInEachQuery.size();
    }


    private List<Integer> instancesForQuery(int queryId){
        return instanceIdsInEachQuery.get(queryId);
    }

    // calculate gradients for all instances in a query
    private double gradientForInstanceInQuery(double[] predictedScores, double[] relevance, int dataIndexInQuery){
        double grade = relevance[dataIndexInQuery];
        double score = predictedScores[dataIndexInQuery];
        double gradient = 0;
        //todo times ndcg delta
        for (int i=0;i<relevance.length;i++){
            if (grade>relevance[i]){
                gradient += 1.0/(1+Math.exp(score - predictedScores[i]));
            }

            if (grade< relevance[i]){
                gradient -= 1.0/(1+Math.exp(predictedScores[i]- score));
            }
        }
        return gradient;
    }

    // calculate gradients for all instances in a query
    private double[] gradientForQuery(int queryIndex){
        List<Integer> instancesForQuery = instancesForQuery(queryIndex);
        double[] predictedScores = instancesForQuery.stream().mapToDouble(i->scoreMatrix.getScoresForData(i)[0]).toArray();
        double[] relevance = instancesForQuery.stream().mapToDouble(i->relevanceGrades[i]).toArray();
        double[] gradients = new double[instancesForQuery.size()];
        for (int i=0;i<gradients.length;i++){
            gradients[i] = gradientForInstanceInQuery(predictedScores,relevance,i);
        }
        return gradients;
    }

    @Override
    protected void addPriors() {
        //for ranking purpose, the neutral point does not matter
        // if we add a prior, it will simply increase the score for all points
    }

    @Override
    protected double[] gradient(int ensembleIndex) {
        double[] gradients = new double[dataSet.getNumDataPoints()];
        IntStream.range(0, numQueries).parallel()
                .forEach(q->{
                    double[] queryGradients = gradientForQuery(q);
                    List<Integer> instancesInQuery = instancesForQuery(q);
                    for (int i=0;i<instancesInQuery.size();i++){
                        int globalIndex = instancesInQuery.get(i);
                        gradients[globalIndex] = queryGradients[i];
                    }
                });
        return gradients;
    }

    @Override
    protected void initializeOthers() {

    }

    @Override
    protected void updateOthers() {

    }
}
