package edu.neu.ccs.pyramid.core.eval;

import com.fasterxml.jackson.core.JsonGenerator;
import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.JsonSerializer;
import com.fasterxml.jackson.databind.SerializerProvider;
import com.fasterxml.jackson.databind.annotation.JsonSerialize;
import edu.neu.ccs.pyramid.core.dataset.MultiLabelClfDataSet;
import edu.neu.ccs.pyramid.core.multilabel_classification.MultiLabelClassifier;

import java.io.IOException;

/**
 * all multi-label measures
 * Created by chengli on 3/24/16.
 */
@JsonSerialize(using = MLMeasures.Serializer.class)
public class MLMeasures {
    private MLConfusionMatrix mlConfusionMatrix;
    private InstanceAverage instanceAverage;
    private MacroAverage macroAverage;
    private MicroAverage microAverage;


    public MLMeasures(MultiLabelClassifier classifier, MultiLabelClfDataSet dataSet){
        this.mlConfusionMatrix = new MLConfusionMatrix(classifier,dataSet);
        this.instanceAverage = new InstanceAverage(mlConfusionMatrix);
        this.macroAverage = new MacroAverage(mlConfusionMatrix);
        this.microAverage = new MicroAverage(mlConfusionMatrix);
    }

    public MLConfusionMatrix getMlConfusionMatrix() {
        return mlConfusionMatrix;
    }

    public InstanceAverage getInstanceAverage() {
        return instanceAverage;
    }

    public MicroAverage getMicroAverage() {
        return microAverage;
    }

    public MacroAverage getMacroAverage() {
        return macroAverage;
    }

    @Override
    public String toString() {
        final StringBuilder sb = new StringBuilder();
        sb.append(instanceAverage);
        sb.append(macroAverage);
        sb.append(microAverage);
        return sb.toString();
    }

    public static class Serializer extends JsonSerializer<MLMeasures> {
        @Override
        public void serialize(MLMeasures mlMeasures, JsonGenerator jsonGenerator,
                              SerializerProvider serializerProvider) throws IOException , JsonProcessingException {

            jsonGenerator.writeStartObject();
            jsonGenerator.writeNumberField("instance subset accuracy",mlMeasures.instanceAverage.getAccuracy());
            jsonGenerator.writeNumberField("instance overlap",mlMeasures.instanceAverage.getOverlap());
            jsonGenerator.writeNumberField("instance F1",mlMeasures.instanceAverage.getF1());
            jsonGenerator.writeNumberField("instance Hamming loss",mlMeasures.instanceAverage.getHammingLoss());
            jsonGenerator.writeNumberField("instance precision",mlMeasures.instanceAverage.getPrecision());
            jsonGenerator.writeNumberField("instance recall",mlMeasures.instanceAverage.getRecall());

            jsonGenerator.writeNumberField("macro overlap",mlMeasures.macroAverage.getOverlap());
            jsonGenerator.writeNumberField("macro F1",mlMeasures.macroAverage.getF1());
            jsonGenerator.writeNumberField("macro Hamming loss",mlMeasures.macroAverage.getHammingLoss());
            jsonGenerator.writeNumberField("macro precision",mlMeasures.macroAverage.getPrecision());
            jsonGenerator.writeNumberField("macro recall",mlMeasures.macroAverage.getRecall());

            jsonGenerator.writeNumberField("micro overlap",mlMeasures.microAverage.getOverlap());
            jsonGenerator.writeNumberField("micro F1",mlMeasures.microAverage.getF1());
            jsonGenerator.writeNumberField("micro Hamming loss",mlMeasures.microAverage.getHammingLoss());
            jsonGenerator.writeNumberField("micro precision",mlMeasures.microAverage.getPrecision());
            jsonGenerator.writeNumberField("micro recall",mlMeasures.microAverage.getRecall());
            jsonGenerator.writeEndObject();
        }
    }
}
