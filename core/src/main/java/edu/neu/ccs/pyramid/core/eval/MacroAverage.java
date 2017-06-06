package edu.neu.ccs.pyramid.core.eval;

import com.fasterxml.jackson.core.JsonGenerator;
import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.JsonSerializer;
import com.fasterxml.jackson.databind.SerializerProvider;
import com.fasterxml.jackson.databind.annotation.JsonSerialize;
import edu.neu.ccs.pyramid.core.dataset.LabelTranslator;

import java.io.IOException;
import java.util.Arrays;
import java.util.stream.IntStream;

/**
 * Based on
 * Koyejo, Oluwasanmi O., et al. "Consistent Multilabel Classification."
 * Advances in Neural Information Processing Systems. 2015.
 * Created by chengli on 3/24/16.
 */
@JsonSerialize(using = MacroAverage.Serializer.class)
public class MacroAverage {
    private int numClasses;
    private double f1;
    private double overlap;
    private double precision;
    private double recall;
    private double hammingLoss;
    private double binaryAccuracy;
    // per class counts
    private int[] labelWiseTP;
    private int[] labelWiseTN;
    private int[] labelWiseFP;
    private int[] labelWiseFN;
    // per class measures
    private double[] labelWisePrecision;
    private double[] labelWiseRecall;
    private double[] labelWiseOverlap;
    private double[] labelWiseF1;
    private double[] labelWiseHammingLoss;
    private double[] labelWiseAccuracy;

    private LabelTranslator labelTranslator;


    public MacroAverage(MLConfusionMatrix confusionMatrix) {
        this.numClasses = confusionMatrix.getNumClasses();
        int numDataPoints = confusionMatrix.getNumDataPoints();
        MLConfusionMatrix.Entry[][] entries = confusionMatrix.getEntries();
        this.labelWiseTP = new int[numClasses];
        this.labelWiseTN = new int[numClasses];
        this.labelWiseFP = new int[numClasses];
        this.labelWiseFN = new int[numClasses];

        this.labelWisePrecision = new double[numClasses];
        this.labelWiseRecall = new double[numClasses];
        this.labelWiseOverlap = new double[numClasses];
        this.labelWiseF1 = new double[numClasses];
        this.labelWiseHammingLoss = new double[numClasses];
        this.labelWiseAccuracy = new double[numClasses];


        IntStream.range(0,numClasses).parallel().forEach(l->{
            for (int i=0;i<numDataPoints;i++){
                MLConfusionMatrix.Entry entry = entries[i][l];
                switch (entry){
                    case TP:
                        labelWiseTP[l] += 1;
                        break;
                    case FP:
                        labelWiseFP[l] += 1;
                        break;
                    case TN:
                        labelWiseTN[l] += 1;
                        break;
                    case FN:
                        labelWiseFN[l] += 1;
                        break;
                }
            }
            double tp = ((double) labelWiseTP[l])/numDataPoints;
            double tn = ((double) labelWiseTN[l])/numDataPoints;
            double fp = ((double) labelWiseFP[l])/numDataPoints;
            double fn = ((double) labelWiseFN[l])/numDataPoints;

            labelWisePrecision[l] = Precision.precision(tp,fp);
            labelWiseRecall[l] = Recall.recall(tp,fn);
            labelWiseF1[l] = FMeasure.f1(tp,fp,fn);
            labelWiseOverlap[l] = Overlap.overlap(tp,fp,fn);
            labelWiseHammingLoss[l] = HammingLoss.hammingLoss(tp,tn,
                    fp,fn);
            labelWiseAccuracy[l] = tp+tn;
        });

        precision = Arrays.stream(labelWisePrecision).average().getAsDouble();

        recall = Arrays.stream(labelWiseRecall).average().getAsDouble();

        f1 = Arrays.stream(labelWiseF1).average().getAsDouble();

        overlap = Arrays.stream(labelWiseOverlap).average().getAsDouble();

        hammingLoss = Arrays.stream(labelWiseHammingLoss).average().getAsDouble();

        binaryAccuracy = Arrays.stream(labelWiseAccuracy).average().getAsDouble();

        this.labelTranslator = LabelTranslator.newDefaultLabelTranslator(numClasses);
    }

    public void setLabelTranslator(LabelTranslator labelTranslator) {
        this.labelTranslator = labelTranslator;
    }

    public double getF1() {
        return f1;
    }

    public double getOverlap() {
        return overlap;
    }

    public double getPrecision() {
        return precision;
    }

    public double getRecall() {
        return recall;
    }

    public double getHammingLoss() {
        return hammingLoss;
    }

    public int[] getLabelWiseTP() {
        return labelWiseTP;
    }

    public int[] getLabelWiseTN() {
        return labelWiseTN;
    }

    public int[] getLabelWiseFP() {
        return labelWiseFP;
    }

    public int[] getLabelWiseFN() {
        return labelWiseFN;
    }

    public double[] getLabelWisePrecision() {
        return labelWisePrecision;
    }

    public double[] getLabelWiseRecall() {
        return labelWiseRecall;
    }

    public double[] getLabelWiseOverlap() {
        return labelWiseOverlap;
    }

    public double[] getLabelWiseF1() {
        return labelWiseF1;
    }

    public double[] getLabelWiseHammingLoss() {
        return labelWiseHammingLoss;
    }

    public double getBinaryAccuracy() {
        return binaryAccuracy;
    }

    public double[] getLabelWiseAccuracy() {
        return labelWiseAccuracy;
    }

    @Override
    public String toString() {
        final StringBuilder sb = new StringBuilder();
        sb.append("macro overlap = ").append(overlap).append("\n");
        sb.append("macro Hamming loss = ").append(hammingLoss).append("\n");
        sb.append("macro F1 = ").append(f1).append("\n");
        sb.append("macro precision = ").append(precision).append("\n");
        sb.append("macro recall = ").append(recall).append("\n");
        sb.append("macro binary accuracy = ").append(binaryAccuracy).append("\n");

        return sb.toString();
    }

    public static class Serializer extends JsonSerializer<MacroAverage> {
        @Override
        public void serialize(MacroAverage macroAverage, JsonGenerator jsonGenerator, SerializerProvider serializerProvider) throws IOException, JsonProcessingException {

            jsonGenerator.writeStartArray();
            for (int k=0;k<macroAverage.numClasses;k++){
                jsonGenerator.writeStartObject();
                jsonGenerator.writeStringField("label", macroAverage.labelTranslator.toExtLabel(k));
                jsonGenerator.writeNumberField("TP",macroAverage.labelWiseTP[k]);
                jsonGenerator.writeNumberField("TN",macroAverage.labelWiseTN[k]);
                jsonGenerator.writeNumberField("FP",macroAverage.labelWiseFP[k]);
                jsonGenerator.writeNumberField("FN",macroAverage.labelWiseFN[k]);
                jsonGenerator.writeNumberField("precision",macroAverage.labelWisePrecision[k]);
                jsonGenerator.writeNumberField("recall",macroAverage.labelWiseRecall[k]);
                jsonGenerator.writeNumberField("f1",macroAverage.labelWiseF1[k]);
                jsonGenerator.writeNumberField("accuracy",macroAverage.labelWiseAccuracy[k]);
                jsonGenerator.writeEndObject();
            }
            jsonGenerator.writeEndArray();
        }
    }
}
