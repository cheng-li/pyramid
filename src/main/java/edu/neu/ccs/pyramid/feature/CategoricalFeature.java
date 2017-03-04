package edu.neu.ccs.pyramid.feature;

import com.fasterxml.jackson.core.JsonGenerator;
import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.JsonSerializer;
import com.fasterxml.jackson.databind.SerializerProvider;
import com.fasterxml.jackson.databind.annotation.JsonSerialize;
import edu.neu.ccs.pyramid.feature.Feature;

import java.io.IOException;
import java.util.Map;

/**
 * Created by chengli on 3/4/15.
 */
@JsonSerialize(using = CategoricalFeature.Serializer.class)
public class CategoricalFeature extends Feature {
    // red
    private String category;
    private int numCategories;
    // color
    private String variableName;
    private Map<String, Integer> categoryIndexMap;



    public String getCategory() {
        return category;
    }

    public void setCategory(String category) {
        this.category = category;
    }

    public String getVariableName() {
        return variableName;
    }

    public void setVariableName(String variableName) {
        this.variableName = variableName;
    }

    public int getNumCategories() {
        return numCategories;
    }

    public void setNumCategories(int numCategories) {
        this.numCategories = numCategories;
    }

    public Map<String, Integer> getCategoryIndexMap() {
        return categoryIndexMap;
    }

    public void setCategoryIndexMap(Map<String, Integer> categoryIndexMap) {
        this.categoryIndexMap = categoryIndexMap;
    }

    @Override
    public String toString() {
        final StringBuilder sb = new StringBuilder("CategoricalFeature{");
        sb.append(super.toString()).append(", ");
        sb.append("category='").append(category).append('\'');
        sb.append(", numCategories=").append(numCategories);
        sb.append(", variableName='").append(variableName).append('\'');
        sb.append(", categoryIndexMap=").append(categoryIndexMap);
        sb.append('}');
        return sb.toString();
    }


    public static class Serializer extends JsonSerializer<CategoricalFeature> {
        @Override
        public void serialize(CategoricalFeature feature, JsonGenerator jsonGenerator, SerializerProvider serializerProvider) throws IOException, JsonProcessingException {
            jsonGenerator.writeStartObject();
            jsonGenerator.writeStringField("name",feature.name);
            jsonGenerator.writeStringField("type","categorical");
            jsonGenerator.writeStringField("variable",feature.variableName);
            jsonGenerator.writeNumberField("numCategories",feature.numCategories);
            jsonGenerator.writeStringField("category",feature.category);
            jsonGenerator.writeEndObject();
        }
    }
}
