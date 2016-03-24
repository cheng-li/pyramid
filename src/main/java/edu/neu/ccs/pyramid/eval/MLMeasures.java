package edu.neu.ccs.pyramid.eval;

import com.fasterxml.jackson.core.JsonGenerator;
import com.fasterxml.jackson.databind.JsonSerializer;
import com.fasterxml.jackson.databind.SerializerProvider;
import com.fasterxml.jackson.databind.annotation.JsonSerialize;
import edu.neu.ccs.pyramid.dataset.MultiLabelClfDataSet;
import edu.neu.ccs.pyramid.multilabel_classification.MultiLabelClassifier;

import java.io.IOException;

/**
 * all multi-label measures
 * Created by chengli on 3/24/16.
 */
@JsonSerialize(using = MLMeasures.Serializer.class)
public class MLMeasures {
    private MLConfusionMatrix mlConfusionMatrix;
    private InstanceAverage instanceAverage;
    private MicroAverage microAverage;

    public MLMeasures(MultiLabelClassifier classifier, MultiLabelClfDataSet dataSet){
        this.mlConfusionMatrix = new MLConfusionMatrix(classifier,dataSet);
        this.instanceAverage = new InstanceAverage(mlConfusionMatrix);
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

    @Override
    public String toString() {
        final StringBuilder sb = new StringBuilder("MLMeasures{");
        sb.append("mlConfusionMatrix=").append(mlConfusionMatrix);
        sb.append(", instanceAverage=").append(instanceAverage);
        sb.append(", microAverage=").append(microAverage);
        sb.append('}');
        return sb.toString();
    }

    public static class Serializer extends JsonSerializer<MLMeasures> {
        @Override
        public void serialize(MLMeasures mlMeasures, JsonGenerator jsonGenerator,
                              SerializerProvider serializerProvider) throws IOException {

            jsonGenerator.writeStartObject();
            jsonGenerator.writeNumberField("instance subset accuracy",mlMeasures.instanceAverage.getAccuracy());
            jsonGenerator.writeNumberField("instance overlap",mlMeasures.instanceAverage.getOverlap());
            jsonGenerator.writeNumberField("instance F1",mlMeasures.instanceAverage.getF1());
            jsonGenerator.writeNumberField("instance Hamming loss",mlMeasures.instanceAverage.getHammingLoss());
            jsonGenerator.writeNumberField("instance precision",mlMeasures.instanceAverage.getPrecision());
            jsonGenerator.writeNumberField("instance recall",mlMeasures.instanceAverage.getRecall());
            jsonGenerator.writeNumberField("micro overlap",mlMeasures.microAverage.getOverlap());
            jsonGenerator.writeNumberField("micro F1",mlMeasures.microAverage.getF1());
            jsonGenerator.writeNumberField("micro Hamming loss",mlMeasures.microAverage.getHammingLoss());
            jsonGenerator.writeNumberField("micro precision",mlMeasures.microAverage.getPrecision());
            jsonGenerator.writeNumberField("micro recall",mlMeasures.microAverage.getRecall());
            jsonGenerator.writeEndObject();
        }
    }
}
