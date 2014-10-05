package edu.neu.ccs.pyramid.configuration;

import java.io.*;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Properties;

/**
 * Created by chengli on 8/11/14.
 */
public class Config {
    private Properties properties;

    public Config(String configFile) {
        this.properties = new Properties();
        try {
            properties.load(new FileInputStream(configFile));
        } catch (IOException e) {
            e.printStackTrace();
        }
//        System.out.println("config file "+configFile+" loaded.");
    }

    public Config(File configFile) {
        this.properties = new Properties();
        try {
            properties.load(new FileInputStream(configFile));
        } catch (IOException e) {
            e.printStackTrace();
        }

    }

    public Config(){
        this.properties = new Properties();
    }

    public void setInt(String key, int value){
        this.properties.setProperty(key,""+value);
    }

    public void setDouble(String key, double value){
        this.properties.setProperty(key,""+value);
    }

    public void setBoolean(String key, boolean value){
        this.properties.setProperty(key,""+value);
    }

    public void setString(String key, String value){
        this.properties.setProperty(key,value);
    }

    public int getInt(String key){
        if (!containsKey(key)){
            throw new IllegalArgumentException("config does not contain the key "+key);
        }
        return Integer.parseInt(this.properties.getProperty(key));
    }

    public double getDouble(String key){
        if (!containsKey(key)){
            throw new IllegalArgumentException("config does not contain the key "+key);
        }
        return Double.parseDouble(this.properties.getProperty(key));
    }

    public boolean getBoolean(String key){
        if (!containsKey(key)){
            throw new IllegalArgumentException("config does not contain the key "+key);
        }
        return Boolean.parseBoolean(this.properties.getProperty(key));
    }

    public String getString(String key){
        if (!containsKey(key)){
            throw new IllegalArgumentException("config does not contain the key "+key);
        }
        return this.properties.getProperty(key);
    }

    public boolean containsKey(String key){
        return this.properties.containsKey(key);
    }

    public void store(Writer writer) throws Exception{
        this.properties.store(writer,"");
    }

    public void store(File file) throws Exception{
        try (FileWriter fileWriter = new FileWriter(file);
             BufferedWriter bufferedWriter = new BufferedWriter(fileWriter)
        ) {
            this.store(bufferedWriter);
        }
    }

    /**
     * copy one key-value pair from src to des
     * @param src
     * @param des
     * @param key
     */
    public static void copy(Config src, Config des, String key){
        des.setString(key,src.getString(key));
    }

    /**
     * copy several key-value pairs from src to des
     * @param src
     * @param des
     * @param keys
     */
    public static void copy(Config src, Config des, String[] keys){
        for (int i=0;i<keys.length;i++){
            String key = keys[i];
            copy(src,des,key);
        }
    }


    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder();
        List<String> list = new ArrayList<>();
        for (String key: this.properties.stringPropertyNames()){
            list.add(key);
        }
        Collections.sort(list);

        for (String key: list){
            sb.append(key).append("=").append(this.properties.getProperty(key));
            sb.append("\n");
        }
        return sb.toString();
    }
}
