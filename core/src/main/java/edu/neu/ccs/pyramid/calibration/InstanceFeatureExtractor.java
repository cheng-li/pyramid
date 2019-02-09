package edu.neu.ccs.pyramid.calibration;

import edu.neu.ccs.pyramid.feature.Feature;
import edu.neu.ccs.pyramid.feature.FeatureList;
import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.Vector;

import java.util.ArrayList;
import java.util.List;
import java.util.regex.Pattern;

public class InstanceFeatureExtractor implements PredictionFeatureExtractor{
    private List<Integer> featureIndices;
    private FeatureList featureList;

    public InstanceFeatureExtractor(String featureIndices, FeatureList featureList) {
        this.featureIndices = new ArrayList<>();
        String[] split = featureIndices.replace(" ","").split(",");
        for (String range: split){
            if (range.contains("-")){
                int start = Integer.parseInt(range.split(Pattern.quote("-"))[0]);
                int end = Integer.parseInt(range.split(Pattern.quote("-"))[1]);
                for (int i=start;i<=end;i++){
                    this.featureIndices.add(i);
                }
            } else {
                this.featureIndices.add(Integer.parseInt(range));
            }
        }
        this.featureList = featureList;
    }

    public InstanceFeatureExtractor(List<Integer> featureIndices, FeatureList featureList) {
        this.featureIndices = featureIndices;
        this.featureList = featureList;
    }

    @Override
    public Vector extractFeatures(PredictionCandidate predictionCandidate) {
        int size = featureIndices.size();
        Vector vector = new DenseVector(size);
        for (int j=0;j<size;j++){
            int index = featureIndices.get(j);
            vector.set(j,predictionCandidate.x.get(index));
        }
        return vector;
    }

    @Override
    public int[] featureMonotonicity() {
        return new int[featureIndices.size()];
    }

    @Override
    public List<Feature> getNames() {
        List<Feature> features = new ArrayList<>();
        for (int i: featureIndices){
            features.add(featureList.get(i));
        }
        return features;
    }

    @Override
    public String toString() {
        final StringBuilder sb = new StringBuilder("InstanceFeatureExtractor{");
        sb.append("featureIndices=").append(featureIndices);
        sb.append(", featureList=").append(featureList);
        sb.append('}');
        return sb.toString();
    }
}
