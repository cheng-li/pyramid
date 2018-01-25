package edu.neu.ccs.pyramid.feature;

import com.fasterxml.jackson.core.JsonGenerator;
import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.JsonSerializer;
import com.fasterxml.jackson.databind.SerializerProvider;
import com.fasterxml.jackson.databind.annotation.JsonSerialize;

import java.io.IOException;
import java.util.List;

/**
 * Created by chengli on 10/6/16.
 */
@JsonSerialize(using = CodeDescription.Serializer.class)
public class CodeDescription extends Feature {
    private static final long serialVersionUID = 1L;
    private List<String> description;
    private int percentage;
    private String field;

    public CodeDescription(List<String> description, int percentage, String field) {
        this.description = description;
        this.percentage = percentage;
        this.field = field;
    }

    public List<String> getDescription() {
        return description;
    }

    public int getPercentage() {
        return percentage;
    }

    public String getField() {
        return field;
    }

    @Override
    public String toString() {
        final StringBuilder sb = new StringBuilder("CodeDescription{");
        sb.append("description=").append(description);
        sb.append(", percentage=").append(percentage);
        sb.append(", field='").append(field).append('\'');
        sb.append('}');
        return sb.toString();
    }

    public static class Serializer extends JsonSerializer<CodeDescription> {
        @Override
        public void serialize(CodeDescription feature, JsonGenerator jsonGenerator, SerializerProvider serializerProvider) throws IOException, JsonProcessingException {
            StringBuilder stringBuilder = new StringBuilder();
            for (String term: feature.description){
                stringBuilder.append(term).append(" ");
            }
            jsonGenerator.writeStartObject();
            jsonGenerator.writeNumberField("index", feature.index);
            jsonGenerator.writeStringField("type","code description");
            jsonGenerator.writeStringField("field",feature.field);
            jsonGenerator.writeStringField("description",stringBuilder.toString());
            jsonGenerator.writeNumberField("minimum should match percent",feature.percentage);

            jsonGenerator.writeEndObject();
        }
    }
}
