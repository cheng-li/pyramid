/**
 *******************************************************************************
 * Copyright by Bishwajeet Dey.
 * All rights reserved.
 *******************************************************************************/
package edu.neu.ccs.pyramid.visualization;

import java.io.File;
import java.io.IOException;
import java.util.Map;

import org.elasticsearch.client.Client;
import org.elasticsearch.common.settings.Settings;
import org.elasticsearch.transport.client.PreBuiltTransportClient;

import com.fasterxml.jackson.core.type.TypeReference;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.google.common.base.Preconditions;

import edu.neu.ccs.pyramid.configuration.Config;

/**
 * Description
 * @author <a href="mailto:deyb@ccs.neu.edu">Bishwajeet Dey</a>
 *
 * @version 1.0.0
 */
public class VisualizerConfig {
    private final File inputFile;
    
    private final File classFile;
    
    private final int numClassesInModel;
    
    private final String esIndexName;
    
    private final String ngramFields;
    
    private final File resourcesDir;
    
    private final Client client;
    
    public VisualizerConfig(final Config config) throws IOException {
        Preconditions.checkNotNull(config);
        
        final ObjectMapper mapper = new ObjectMapper();
        
        this.inputFile = new File(config.getString("input.file"));
        checkFileExists(inputFile);
    
        final File dataConfigFile = new File(inputFile.getParentFile(), config.getString("data.config.name"));
        final Map<String, Object> dataConfig =  mapper.readValue(dataConfigFile, new TypeReference<Map<String, Object>>() {
        });
        this.esIndexName = dataConfig.get("index.indexName") == null ? "ohsumed_20000" : (String) dataConfig.get("index.indexName");
        this.ngramFields = (String) dataConfig.get("index.ngramExtractionFields");
        
        final File dataInfoFile   = new File(inputFile.getParentFile(), config.getString("data.info.name"));
        this.numClassesInModel = mapper.readTree(dataInfoFile).path("numClassesInModel").asInt(23);
        
        //Not used anywhere        
        //final File modelConfigFile = new File(inputFile.getParentFile(), config.getString("model.config.name"));
        
        this.classFile = new File(config.getString("class.file"));
        
        this.resourcesDir = new File(config.getString("resources.dir"));
        
        checkFileExists(classFile);
        Settings settings = Settings.builder()
                .put("cluster.name", "fijielasticsearch").build();
        this.client =    new PreBuiltTransportClient(settings);
        /*
        this.client = new TransportClient()
                .addTransportAddress(new InetSocketTransportAddress("127.0.0.1", 9200));*/
    }
       
    /**
     * Throw up {@link IllegalStateException} if file is not a file(ie a dir, symlink etc)
     * @param file
     * @throws IOException
     */
    private void checkFileExists(File file) throws IOException {
        if (!file.isFile()) {
            throw new IllegalStateException(file.getCanonicalPath() + "is not a file");
        }
    }

    /**
     * @return the inputFile
     */
    public File getInputFile() {
        return inputFile;
    }

    /**
     * @return the classFile
     */
    public File getClassFile() {
        return classFile;
    }

    /**
     * @return the numClassesInModel
     */
    public int getNumClassesInModel() {
        return numClassesInModel;
    }

    /**
     * @return the esIndexName
     */
    public String getEsIndexName() {
        return esIndexName;
    }
    
    public File getInputFileBaseDir() {
        return inputFile.getParentFile();
    }

    /**
     * @return the ngramFields
     */
    public String getNgramFields() {
        return ngramFields;
    }

    /**
     * @return the resourcesDir
     */
    public File getResourcesDir() {
        return resourcesDir;
    }

    /**
     * @return the client
     */
    public Client getClient() {
        return client;
    }
}
