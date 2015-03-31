package edu.neu.ccs.pyramid.feature;

import com.fasterxml.jackson.core.JsonGenerator;
import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.JsonSerializer;
import com.fasterxml.jackson.databind.SerializerProvider;
import com.fasterxml.jackson.databind.annotation.JsonSerialize;

import java.io.IOException;
import java.util.List;

/**
 * Created by chengli on 3/7/15.
 */
@JsonSerialize(using = Ngram.Serializer.class)
public class Ngram extends Feature {
    private static final long serialVersionUID = 3L;
    private String ngram = "unknown";
    private String field = "unknown";
    private int slop;
    private boolean inOrder=true;



    public String getNgram() {
        return ngram;
    }

    public String[] getTerms(){
        return ngram.split(" ");
    }

    public void setNgram(String ngram) {
        this.ngram = ngram;
    }

    public String getField() {
        return field;
    }

    public void setField(String field) {
        this.field = field;
    }

    public int getSlop() {
        return slop;
    }

    public void setSlop(int slop) {
        this.slop = slop;
    }

    public int getN(){
        return ngram.split(" ").length;
    }

    public boolean isInOrder() {
        return inOrder;
    }

    public void setInOrder(boolean inOrder) {
        this.inOrder = inOrder;
    }

    public static String toNgramString(List<String> terms){
        String str = "";
        for (int i=0;i<terms.size();i++){
            str = str.concat(terms.get(i));
            if (i!=terms.size()-1){
                str = str.concat(" ");
            }
        }
        return str;
    }

    @Override
    public String toString() {
        final StringBuilder sb = new StringBuilder("Ngram{");
        if (indexAssigned){
            sb.append("index=").append(index).append(",");
        }
        sb.append("name=").append(name);
        sb.append(", ngram=").append(ngram);
        sb.append(", field=").append(field);
        sb.append(", slop=").append(slop);
        sb.append(", inOrder=").append(inOrder);
        sb.append(", settings=").append(settings);
        sb.append('}');
        return sb.toString();
    }

    public static class Serializer extends JsonSerializer<Ngram> {
        @Override
        public void serialize(Ngram feature, JsonGenerator jsonGenerator, SerializerProvider serializerProvider) throws IOException, JsonProcessingException {
            jsonGenerator.writeStartObject();
            jsonGenerator.writeNumberField("index", feature.index);
            jsonGenerator.writeStringField("name", feature.name);
            jsonGenerator.writeStringField("ngram",feature.ngram);
            jsonGenerator.writeStringField("field",feature.field);
            jsonGenerator.writeNumberField("slop", feature.slop);
            jsonGenerator.writeEndObject();
        }
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;
        if (!super.equals(o)) return false;

        Ngram ngram1 = (Ngram) o;

        if (inOrder != ngram1.inOrder) return false;
        if (slop != ngram1.slop) return false;
        if (!field.equals(ngram1.field)) return false;
        if (!ngram.equals(ngram1.ngram)) return false;

        return true;
    }

    @Override
    public int hashCode() {
        int result = super.hashCode();
        result = 31 * result + ngram.hashCode();
        result = 31 * result + field.hashCode();
        result = 31 * result + slop;
        result = 31 * result + (inOrder ? 1 : 0);
        return result;
    }
}
