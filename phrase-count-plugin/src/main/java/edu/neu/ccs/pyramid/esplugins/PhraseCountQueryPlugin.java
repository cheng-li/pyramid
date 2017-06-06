package edu.neu.ccs.pyramid.esplugins;

import org.elasticsearch.plugins.Plugin;
import org.elasticsearch.plugins.SearchPlugin;

import java.util.LinkedList;
import java.util.List;

/**
 * Created by maoqiuzi on 5/23/17.
 */
public class PhraseCountQueryPlugin extends Plugin implements SearchPlugin {
    @Override
    public List<QuerySpec<?>> getQueries() {
        List<QuerySpec<?>> list = new LinkedList<>();
        list.add(new QuerySpec<>(PhraseCountQueryBuilder.NAME, PhraseCountQueryBuilder::new, PhraseCountQueryBuilder::fromXContent));
        return list;
    }
}
