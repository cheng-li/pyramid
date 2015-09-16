package edu.neu.ccs.pyramid.eval;

import com.fasterxml.jackson.core.JsonGenerator;
import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.JsonSerializer;
import com.fasterxml.jackson.databind.SerializerProvider;
import com.fasterxml.jackson.databind.annotation.JsonSerialize;
import edu.neu.ccs.pyramid.dataset.LabelTranslator;
import edu.neu.ccs.pyramid.dataset.MultiLabel;
import edu.neu.ccs.pyramid.dataset.MultiLabelClfDataSet;

import java.io.IOException;

/**
 * Created by Rainicy on 8/11/15.
 */
@JsonSerialize(using = LabelBasedMeasures.Serializer.class)
public class LabelBasedMeasures {

    /**
     * number of unique labels.
     */
    protected int numLabels;

    /**
     * for each label(index of the array),
     * the number of true positives.
     */
    protected int[] truePositives;

    /**
     * for each label(index of the array),
     * the number of true negatives.
     */
    protected int[] trueNegatives;

    /**
     * for each label(index of the array),
     * the number of false positives
     */
    protected int[] falsePositives;

    /**
     * for each label(index of the array),
     * the number of false negatives.
     */
    protected int[] falseNegatives;

    protected int numDataPoitns;

    protected LabelTranslator labelTranslator;


    public LabelBasedMeasures(MultiLabelClfDataSet dataSet, MultiLabel[] prediction) {
        this(dataSet.getNumClasses());
        this.labelTranslator = dataSet.getLabelTranslator();
        update(dataSet.getMultiLabels(),prediction);
    }

    /**
     * Construct function: initialize each variables.
     * @param numLabels
     */
    public LabelBasedMeasures(int numLabels) {
        if (numLabels == 0) {
            throw new RuntimeException("initialization with zero label.");
        }
        this.numLabels = numLabels;

        truePositives = new int[numLabels];
        falsePositives = new int[numLabels];
        trueNegatives = new int[numLabels];
        falseNegatives = new int[numLabels];
        this.numDataPoitns = 0;
        this.labelTranslator = LabelTranslator.newDefaultLabelTranslator(numLabels);
    }

    public double precision(int classIndex){
        return ConfusionMatrixMeasures.precision(truePositives[classIndex],falsePositives[classIndex]);
    }

    public double recall(int classIndex){
        return ConfusionMatrixMeasures.recall(truePositives[classIndex],falseNegatives[classIndex]);
    }

    public double f1(int classIndex){
        double precision = precision(classIndex);
        double recall = recall(classIndex);
        return FMeasure.f1(precision,recall);
    }

    public double accuracy(int classIndex){
        return ConfusionMatrixMeasures.accuracy(truePositives[classIndex], trueNegatives[classIndex], falsePositives[classIndex], falseNegatives[classIndex]);
    }


    /**
     * update the confusion matrix by given one sample
     * ground truth and prediction.
     * @param label ground truth
     * @param prediction predictions
     */
    public void update(MultiLabel label, MultiLabel prediction) {

        for (int i=0; i<numLabels; i++) {
            boolean actual = label.matchClass(i);
            boolean predicted = prediction.matchClass(i);

            if (actual) {
                if (predicted) {
                    truePositives[i]++;
                } else {
                    falseNegatives[i]++;
                }
            } else {
                if (predicted) {
                    falsePositives[i]++;
                } else {
                    trueNegatives[i]++;
                }
            }
            numDataPoitns ++;
        }
    }

    /**
     * update the confusion matrix by given an array of ground truth and
     * predictions.
     * @param labels ground truth array
     * @param predictions prediction array
     */
    public void update(MultiLabel[] labels, MultiLabel[] predictions) {

        if (labels.length == 0) {
            throw new RuntimeException("Empty given ground truth.");
        }
        if (labels.length != predictions.length) {
            throw new RuntimeException("The lengths of ground truth and predictions should" +
                    "be the same.");
        }

        for (int i=0; i<labels.length; ++i) {
            update(labels[i], predictions[i]);
        }
    }

    public static class Serializer extends JsonSerializer<LabelBasedMeasures> {
        @Override
        public void serialize(LabelBasedMeasures labelBasedMeasures, JsonGenerator jsonGenerator, SerializerProvider serializerProvider) throws IOException, JsonProcessingException {

            LabelTranslator labelTranslator = labelBasedMeasures.labelTranslator;
            jsonGenerator.writeStartArray();
            for (int k=0;k<labelBasedMeasures.numLabels;k++){
                jsonGenerator.writeStartObject();
                jsonGenerator.writeStringField("label", labelTranslator.toExtLabel(k));
                jsonGenerator.writeNumberField("TP",labelBasedMeasures.truePositives[k]);
                jsonGenerator.writeNumberField("TN",labelBasedMeasures.trueNegatives[k]);
                jsonGenerator.writeNumberField("FP",labelBasedMeasures.falsePositives[k]);
                jsonGenerator.writeNumberField("FN",labelBasedMeasures.falseNegatives[k]);
                jsonGenerator.writeNumberField("precision",labelBasedMeasures.precision(k));
                jsonGenerator.writeNumberField("recall",labelBasedMeasures.recall(k));
                jsonGenerator.writeNumberField("f1",labelBasedMeasures.f1(k));
                jsonGenerator.writeNumberField("accuracy",labelBasedMeasures.accuracy(k));
                jsonGenerator.writeEndObject();
            }
            jsonGenerator.writeEndArray();
        }
    }



}
