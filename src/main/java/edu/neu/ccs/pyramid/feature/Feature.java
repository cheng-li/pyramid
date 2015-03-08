package edu.neu.ccs.pyramid.feature;

import java.io.Serializable;
import java.util.HashMap;
import java.util.Map;

/**
 * Created by chengli on 3/4/15.
 */
public class Feature implements Serializable{
    private static final long serialVersionUID = 1L;
    private int index;
    private String name = "unknown";
    private Map<String,String> settings;

    public Feature() {
        this.settings = new HashMap<>();
    }

    public String getName() {
        return name;
    }

    public void setName(String name) {
        this.name = name;
    }

    public int getIndex() {
        return index;
    }

    public void setIndex(int index) {
        this.index = index;
    }

    public Map<String, String> getSettings() {
        return settings;
    }

    @Override
    public String toString() {
        return "Feature{" +
                "index=" + index +
                ", name='" + name + '\'' +
                ", settings=" + settings +
                '}';
    }
}
