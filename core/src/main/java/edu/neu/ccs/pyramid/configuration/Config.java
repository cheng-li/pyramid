package edu.neu.ccs.pyramid.configuration;

import com.fasterxml.jackson.core.JsonGenerator;
import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.JsonSerializer;
import com.fasterxml.jackson.databind.SerializerProvider;
import com.fasterxml.jackson.databind.annotation.JsonSerialize;
import edu.neu.ccs.pyramid.Version;
import org.apache.commons.io.FileUtils;

import java.io.*;
import java.util.*;

/**
 * Created by chengli on 8/11/14.
 */
@JsonSerialize(using = Config.Serializer.class)
public class Config implements Serializable{
    private static final long serialVersionUID = 1L;
    private Properties properties;

    public Config(String configFile) {
        this(new File(configFile));
    }

    public Config(File configFile) {
        this.properties = new Properties();
        try( FileInputStream fileInputStream = new FileInputStream(configFile)){
            properties.load(fileInputStream);
        } catch (IOException e) {
            e.printStackTrace();
        }
        this.setString("pyramid.version", Version.getVersion());
    }

    public Config(){
        this.properties = new Properties();
        this.setString("pyramid.version", Version.getVersion());
    }


    public static Config newConfigFromString(String s) {
        Config config = new Config();
        config.properties = new Properties();
        try {
            config.properties.load(new StringReader(s));
        } catch (IOException e) {
            e.printStackTrace();
        }
        return config;
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

    public List<String> getStrings(String key){
        if (!containsKey(key)){
            throw new IllegalArgumentException("config does not contain the key "+key);
        }
        List<String> list = new ArrayList<>();
        String values = this.properties.getProperty(key);
        if (values.equals("")){
            return list;
        }
        String[] valuesSplit = values.split(",");
        for (String valueString: valuesSplit){
            list.add(valueString.trim());
        }
        return list;
    }

    public List<Integer> getIntegers(String key){
        List<String> strings = getStrings(key);
        List<Integer> list = new ArrayList<>();
        for (String valueString: strings){
            list.add(Integer.parseInt(valueString.trim()));
        }
        return list;
    }

    public List<Double> getDoubles(String key){
        List<String> strings = getStrings(key);
        List<Double> list = new ArrayList<>();
        for (String valueString: strings){
            list.add(Double.parseDouble(valueString.trim()));
        }
        return list;
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

    public void storeOrdered(File file) throws Exception{
        String s = this.toString();
        FileUtils.writeStringToFile(file, s);
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

    /**
     * only copy existing keys from the array
     * @param src
     * @param des
     * @param keys
     */
    public static void copyExisting(Config src, Config des, String[] keys){
        for (int i=0;i<keys.length;i++){
            String key = keys[i];
            if (src.containsKey(key)){
                copy(src,des,key);
            }

        }
    }

    /**
     * copy all key-value pairs
     * @param src
     * @param des
     */
    public static void copy(Config src, Config des){
        for (String key: src.getKeys()){
            copy(src,des,key);
        }
    }

    public Set<String> getKeys(){
        return properties.stringPropertyNames();
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


    public static class Serializer extends JsonSerializer<Config>{
        @Override
        public void serialize(Config config, JsonGenerator jsonGenerator, SerializerProvider serializerProvider) throws IOException, JsonProcessingException {
            List<String> list = new ArrayList<>();
            for (String key: config.properties.stringPropertyNames()){
                list.add(key);
            }
            Collections.sort(list);
            jsonGenerator.writeStartObject();
            for (String key: list){
                jsonGenerator.writeStringField(key,config.getString(key));
            }
            jsonGenerator.writeEndObject();
        }
    }
}
