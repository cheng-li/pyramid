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
    private static final long serialVersionUID = 2L;
    private List<String> description;
    private int percentage;
    private String field;
    private String descriptionString;
    private int size;
    private String code;

    public CodeDescription(List<String> description, int percentage, String field) {
        this.description = description;
        this.percentage = percentage;
        this.field = field;
    }

    public CodeDescription(String descriptionString, String field, int size, String code) {
        this.descriptionString = descriptionString;
        this.size = size;
        this.field = field;
        this.code = code;
    }


    public List<String> getDescription() {
        return description;
    }

    public String getDescriptionString(){ return descriptionString;}

    public int getSize(){return size;}

    public int getPercentage() {
        return percentage;
    }

    public String getField() {
        return field;
    }

    public String getCode(){return code;}

    @Override
    public String toString() {
        return "CodeDescription{" +
                "description=" + description +
                ", percentage=" + percentage +
                ", field='" + field + '\'' +
                ", descriptionString='" + descriptionString + '\'' +
                ", size=" + size +
                ", code='" + code + '\'' +
                '}';
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
