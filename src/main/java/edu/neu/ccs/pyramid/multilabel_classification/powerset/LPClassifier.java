package edu.neu.ccs.pyramid.multilabel_classification.powerset;

import edu.neu.ccs.pyramid.classification.Classifier;
import edu.neu.ccs.pyramid.dataset.LabelTranslator;
import edu.neu.ccs.pyramid.dataset.MultiLabel;
import edu.neu.ccs.pyramid.dataset.MultiLabelClfDataSet;
import edu.neu.ccs.pyramid.feature.FeatureList;
import edu.neu.ccs.pyramid.multilabel_classification.MultiLabelClassifier;
import org.apache.mahout.math.Vector;

import java.io.*;
import java.util.HashMap;
import java.util.Map;

/**
 * Created by Rainicy on 12/3/15.
 */
public class LPClassifier implements MultiLabelClassifier, Serializable {


    int numLabels;

    FeatureList featureList;

    LabelTranslator labelTranslator;

    Map<Integer, MultiLabel> IDToML;
    Map<MultiLabel, Integer> MLToID;

    // for multi class
    Classifier.ScoreEstimator estimator;




    public LPClassifier(MultiLabelClfDataSet dataSet) {
        this(dataSet.getNumClasses());
    }

    public LPClassifier(int numLabels) {
        this.numLabels = numLabels;
    }



    @Override
    public int getNumClasses() {
        return this.numLabels;
    }

    @Override
    public MultiLabel predict(Vector vector) {
        int predIndex = estimator.predict(vector);
        return IDToML.get(predIndex);
    }


    public String toString() {
        StringBuilder result = new StringBuilder();

        result.append("featureList: \n" + featureList);
        result.append("labelTranslator: \n" + labelTranslator);
        result.append("Index to MultiLabel: \n" + IDToML.toString());
        result.append("estimator: :\n" + estimator.toString());
        return result.toString();
    }

    @Override
    public void serialize(File file) throws Exception {
        File parent = file.getParentFile();
        if (!parent.exists()) {
            parent.mkdir();
        }
        try (
                FileOutputStream fileOutputStream = new FileOutputStream(file);
                BufferedOutputStream bufferedOutputStream = new BufferedOutputStream(fileOutputStream);
                ObjectOutputStream objectOutputStream = new ObjectOutputStream(bufferedOutputStream);
        ){
            objectOutputStream.writeObject(this);
        }
    }

    @Override
    public void serialize(String file) throws Exception {
        File file1 = new File(file);
        serialize(file1);
    }

    public static LPClassifier deserialize(File file) throws Exception {
        try (
                FileInputStream fileInputStream = new FileInputStream(file);
                BufferedInputStream bufferedInputStream = new BufferedInputStream(fileInputStream);
                ObjectInputStream objectInputStream = new ObjectInputStream(bufferedInputStream);
        ){
            LPClassifier bmmClassifier = (LPClassifier) objectInputStream.readObject();
            return bmmClassifier;
        }
    }

    public static LPClassifier deserialize(String file) throws Exception {
        File file1 = new File(file);
        return deserialize(file1);
    }

    @Override
    public FeatureList getFeatureList() {
        return null;
    }

    @Override
    public LabelTranslator getLabelTranslator() {
        return null;
    }

    public void setIDToML(Map<Integer, MultiLabel> IDToML) {
        this.IDToML = new HashMap<>(IDToML);
    }

    public void setMLToID(Map<MultiLabel, Integer> MLToID) {
        this.MLToID = new HashMap<>(MLToID);
    }

    public void setLabelTranslator(LabelTranslator labelTranslator) {
        this.labelTranslator = labelTranslator;
    }

    public void setFeatureList(FeatureList featureList) {
        this.featureList = featureList;
    }

    public Map<Integer, MultiLabel> getIDToML() {
        return IDToML;
    }

    public Map<MultiLabel, Integer> getMLToID() {
        return MLToID;
    }

    public Classifier.ScoreEstimator getEstimator() {
        return estimator;
    }
}
