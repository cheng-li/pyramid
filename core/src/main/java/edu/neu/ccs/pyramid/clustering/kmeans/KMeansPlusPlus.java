package edu.neu.ccs.pyramid.clustering.kmeans;

import edu.neu.ccs.pyramid.dataset.DataSet;
import edu.neu.ccs.pyramid.util.MathUtil;
import edu.neu.ccs.pyramid.util.Sampling;
import org.apache.commons.math3.distribution.EnumeratedIntegerDistribution;
import org.apache.mahout.math.Vector;

import java.util.ArrayList;
import java.util.List;
import java.util.stream.IntStream;

public class KMeansPlusPlus {
    private int numComponents;
    private List<Vector> centers;
    private DataSet dataSet;
    private double[] distances;


    public KMeansPlusPlus(int numComponents, DataSet dataSet) {
        this.numComponents = numComponents;
        this.dataSet = dataSet;
        this.centers = new ArrayList<>();
        this.distances = new double[dataSet.getNumDataPoints()];
    }

    public void initialize(){
        System.out.println("initialize");
        int dataIndex = Sampling.intUniform(0,dataSet.getNumDataPoints()-1);
        centers.add(dataSet.getRow(dataIndex));
        System.out.println("randomly pick instance "+(dataIndex+1)+" as the initial centroid for cluster "+centers.size());
        while(centers.size()<numComponents){
            updateDistance();
            double sum = MathUtil.arraySum(distances);
            for (int i=0;i<distances.length;i++){
                distances[i] /= sum;
            }
            int[] indices = IntStream.range(0, dataSet.getNumDataPoints()).toArray();
            EnumeratedIntegerDistribution dis = new EnumeratedIntegerDistribution(indices, distances);
            int sample = dis.sample();
            centers.add(dataSet.getRow(sample));
            System.out.println("randomly pick instance "+(sample+1)+" as the initial centroid for cluster "+centers.size());
        }
    }

    public List<Vector> getCenters() {
        return centers;
    }

    private void updateDistance(int i){
        double min = IntStream.range(0, centers.size())
                .mapToDouble(k->KMeans.distance(dataSet.getRow(i),centers.get(k)))
                .min().getAsDouble();
        distances[i] = min*min;
    }

    private void updateDistance(){
        IntStream.range(0, dataSet.getNumDataPoints()).parallel()
                .forEach(i-> updateDistance(i));
    }


}
