# phrase-count-query

Elasticsearch provided several similarity scores by default. However, sometimes one might need to use raw count of phrase for his own purpose. This plugin provides a query that would return the count of some phrase.

## Usage

sample usage in Kibana:
```
GET my_index/_search
{
  "query": {
    "bool": {
      "must": {
        "phrase_count_query": {
          "field": {
            "query": "term1 term2",
            "slop": 3,
            "weighted_count": "true"
          }
        }
      }
    }
  }
}
Explanation:
"phrase_count_query": // the name of this query
"field": // field to query
"query": "term1 term2" // phrases to query
"slop": 3 // the slop of the term match, default 0
"weighted_count": "true" // whether return weighted count. default false, which increment one on every occurance of phrase.
```



## How to compile and install
### compile
- change to the project root directory
- change the elasticsearch.version in pom to reflect the elasticsearch being used. Should support 5.x
- `mvn package` This should generate the package as a zip file(required by elasticsearch) under target/releases/
- If needed to added to local repo, run `mvn install`
### install plugin
- change to elasticsearch install directory
- if already installed, run `./bin/elasticsearch-plugin remove phrase-count-plugin` to remove
- run `./bin/elasticsearch-plugin install file:/path/target-file-name.zip`. The "file:" part is only needed for local zip package.

## Compatibility
Should work for Elasticsearch from 5.2.x to 5.4.x
