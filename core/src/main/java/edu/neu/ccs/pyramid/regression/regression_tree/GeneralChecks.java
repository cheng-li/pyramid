package edu.neu.ccs.pyramid.regression.regression_tree;

import com.fasterxml.jackson.core.JsonGenerator;
import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.JsonSerializer;
import com.fasterxml.jackson.databind.SerializerProvider;
import com.fasterxml.jackson.databind.annotation.JsonSerialize;
import edu.neu.ccs.pyramid.feature.Feature;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

@JsonSerialize(using = GeneralChecks.Serializer.class)
public class GeneralChecks {
    List<Integer> featureIndices;
    List<Feature> features;
    List<Double> thresholds;
    //true=go left
    List<Boolean> directions;



    public GeneralChecks() {
        this.featureIndices = new ArrayList<>();
        this.features = new ArrayList<>();
        this.thresholds = new ArrayList<>();
        this.directions = new ArrayList<>();
    }



    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;

        Checks that = (Checks) o;

        if (!directions.equals(that.directions)) return false;
        if (!featureIndices.equals(that.featureIndices)) return false;
        if (!features.equals(that.features)) return false;
        if (!thresholds.equals(that.thresholds)) return false;

        return true;
    }

    @Override
    public int hashCode() {
        int result = featureIndices.hashCode();
        result = 31 * result + features.hashCode();
        result = 31 * result + thresholds.hashCode();
        result = 31 * result + directions.hashCode();
        return result;
    }

    public Checks copy(){
        Checks copy = new Checks();
        for (int i=0;i<this.featureIndices.size();i++){
            copy.featureIndices.add(this.featureIndices.get(i));
            copy.features.add(this.features.get(i));
            copy.thresholds.add(this.thresholds.get(i));
        }
        return copy;
    }

    public static class Serializer extends JsonSerializer<GeneralChecks> {
        @Override
        public void serialize(GeneralChecks checks, JsonGenerator jsonGenerator,
                              SerializerProvider serializerProvider) throws IOException, JsonProcessingException {
//            jsonGenerator.writeStartObject();
//            jsonGenerator.writeFieldName("checks");
            jsonGenerator.writeStartArray();
            for (int i=0;i< checks.featureIndices.size();i++){

                int featureIndex = checks.featureIndices.get(i);
                Feature feature = checks.features.get(i);
                double threshold = checks.thresholds.get(i);
                boolean direction = checks.directions.get(i);
                jsonGenerator.writeStartObject();

                jsonGenerator.writeObjectField("feature",feature);
                jsonGenerator.writeFieldName("relation");
                if (direction){
                    jsonGenerator.writeString("<=");
                } else {
                    jsonGenerator.writeString(">");
                }
                jsonGenerator.writeNumberField("threshold",threshold);
                jsonGenerator.writeEndObject();
            }
            jsonGenerator.writeEndArray();
//            jsonGenerator.writeEndObject();
        }
    }
}
