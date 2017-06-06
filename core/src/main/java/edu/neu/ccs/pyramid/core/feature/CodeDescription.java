package edu.neu.ccs.pyramid.core.feature;

import java.util.List;

/**
 * Created by chengli on 10/6/16.
 */
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
}
