package edu.neu.ccs.pyramid.clustering.kmeans;

import edu.neu.ccs.pyramid.dataset.DataSet;
import edu.neu.ccs.pyramid.util.ArgMin;
import edu.neu.ccs.pyramid.util.MathUtil;
import edu.neu.ccs.pyramid.util.Sampling;
import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.Vector;

import java.util.Arrays;
import java.util.List;
import java.util.stream.IntStream;

public class KMeans {
    private int numComponents;
    private Vector[] centers;
    private DataSet dataSet;
    private int[] assignments;
    private boolean[] assigned;

    public KMeans(int numComponents, DataSet dataSet) {
        this.numComponents = numComponents;
        this.dataSet = dataSet;
        this.centers = new DenseVector[numComponents];
        this.assignments = new int[dataSet.getNumDataPoints()];
        this.assigned = new boolean[dataSet.getNumDataPoints()];
    }

    public void iterate(){
        updateCenters();
        assign(true);

    }

    public int getNumComponents() {
        return numComponents;
    }

    public Vector[] getCenters() {
        return centers;
    }

    public int[] getAssignments() {
        return assignments;
    }

    private void updateCenters(){
        System.out.println("update cluster centroids");
        IntStream.range(0,numComponents)
                //todo
//                .parallel()
                .forEach(this::updateCenters);
    }

    public void randomInitialize(){
        for (int k=0;k<numComponents;k++){
            int dataIndex = Sampling.intUniform(0,dataSet.getNumDataPoints()-1);
            centers[k] = dataSet.getRow(dataIndex);
        }
        assign(true);
    }

    public void kmeansPlusPlusInitialize(){
        KMeansPlusPlus kMeansPlusPlus = new KMeansPlusPlus(this.numComponents,this.dataSet);
        kMeansPlusPlus.initialize(true);
        List<Vector> c = kMeansPlusPlus.getCenters();
        for (int k=0;k<numComponents;k++){
            centers[k] = c.get(k);
        }
        assign(true);
    }


    public void kmeansPlusPlusInitialize(int numRuns){
        System.out.println("initialize");
        List<Integer> bestIds = null;
        double bestObj = Double.POSITIVE_INFINITY;
        for (int r=0;r<numRuns;r++){
            KMeansPlusPlus kMeansPlusPlus = new KMeansPlusPlus(this.numComponents,this.dataSet);
            kMeansPlusPlus.initialize(false);
            List<Integer> pickedIds = kMeansPlusPlus.getPickedIds();
            for (int k=0;k<numComponents;k++){
                centers[k] = dataSet.getRow(pickedIds.get(k));
            }
            assign(false);
            double obj = objective();
            if (obj<bestObj){
                bestObj = obj;
                bestIds = pickedIds;
            }
        }

        for (int k=0;k<numComponents;k++){
            centers[k] = dataSet.getRow(bestIds.get(k));
            System.out.println("randomly pick instance "+(bestIds.get(k)+1)+" as the initial centroid for cluster "+(k+1));
        }
        Arrays.fill(assigned,false);
        assign(true);

    }

    public double objective(){
        return IntStream.range(0, dataSet.getNumDataPoints()).parallel()
                .mapToDouble(this::objective).sum();
    }
    private double objective(int i){
        return Math.pow(distance(dataSet.getRow(i),centers[assignments[i]]),2);
    }

    private void updateCenters(int k){

        Vector center = new DenseVector(dataSet.getNumFeatures());
        double count = 0;
        for (int i=0;i<dataSet.getNumDataPoints();i++){
            if (assignments[i]==k){
                Vector instance = dataSet.getRow(i);
                for (int j=0;j<instance.size();j++){
                    center.set(j, center.get(j)+instance.get(j));
                }
                count += 1;
            }
        }
        center = center.divide(count);
        centers[k] = center;
        System.out.println("update the centroid of cluster "+(k+1)+" based on "+(int)count+" instances in the cluster");
    }

    private void assign(int i, boolean print){
        int previousAssignment = assignments[i];
        Vector vector = dataSet.getRow(i);
        double[] distances = IntStream.range(0,numComponents).mapToDouble(k->distance(vector, centers[k]))
                .toArray();
        int assignedC =  ArgMin.argMin(distances);
        assignments[i] = assignedC;
        if (print){
            if (assigned[i] && (previousAssignment!=assignedC)){
                System.out.println("assign instance "+(i+1)+" to cluster "+(assignedC+1)+", previously in cluster "+(previousAssignment+1));
            } else {
                System.out.println("assign instance "+(i+1)+" to cluster "+(assignedC+1));
            }
        }


        assigned[i] = true;
    }


    private void assign(boolean print){
        if (print){
            System.out.println("assign each instance to its nearest cluster");
        }

        IntStream.range(0, dataSet.getNumDataPoints())
                //todo
//                .parallel()
                .forEach(i->assign(i,print));
    }

    static double distance(Vector vector1, Vector vector2){
        double sum = 0;
        for (int i=0;i<vector1.size();i++){
            double diff = (vector1.get(i)-vector2.get(i));
            sum += diff*diff;
        }
        return Math.pow(sum,0.5);
    }

}
