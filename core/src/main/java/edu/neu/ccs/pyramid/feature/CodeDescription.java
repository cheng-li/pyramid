package edu.neu.ccs.pyramid.feature;

import java.util.List;

/**
 * Created by chengli on 10/6/16.
 */
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
}
