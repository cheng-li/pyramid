package edu.neu.ccs.pyramid.dataset;

import java.io.Serializable;
import java.util.HashMap;
import java.util.Map;

/**
 * Created by chengli on 3/4/15.
 */
public class Feature implements Serializable{
    private static final long serialVersionUID = 1L;
    private int index;
    private String name;
    private Map<String,String> setting;

    public Feature() {
        this.setting = new HashMap<>();
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

    public Map<String, String> getSetting() {
        return setting;
    }


}
