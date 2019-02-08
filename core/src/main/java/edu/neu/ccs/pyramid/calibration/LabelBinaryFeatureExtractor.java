package edu.neu.ccs.pyramid.calibration;

import edu.neu.ccs.pyramid.dataset.LabelTranslator;
import edu.neu.ccs.pyramid.dataset.MultiLabel;
import edu.neu.ccs.pyramid.feature.Feature;
import org.apache.mahout.math.RandomAccessSparseVector;
import org.apache.mahout.math.Vector;

import java.util.ArrayList;
import java.util.List;

public class LabelBinaryFeatureExtractor implements PredictionFeatureExtractor{
    int numLabelsInModel;
    LabelTranslator labelTranslator;

    public LabelBinaryFeatureExtractor(int numLabelsInModel, LabelTranslator labelTranslator) {
        this.numLabelsInModel = numLabelsInModel;
        this.labelTranslator = labelTranslator;
    }

    @Override
    public Vector extractFeatures(MultiLabel prediction) {
        Vector vector = new RandomAccessSparseVector(numLabelsInModel);
        for (int l: prediction.getMatchedLabels()){
            vector.set(l,1);
        }
        return vector;
    }

    @Override
    public int[] featureMonotonicity() {
        return new int[numLabelsInModel];
    }

    @Override
    public List<Feature> getNames() {

        List<Feature> features = new ArrayList<>();
        for (int l=0;l<numLabelsInModel;l++){
            Feature feature = new Feature();
            feature.setName("label_"+labelTranslator.toExtLabel(l));
            features.add(feature);
        }

        return features;
    }
}
