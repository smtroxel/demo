import copy
import gensim
import spacy
import time
import math
import logging
from gensim.parsing.preprocessing import (
    preprocess_string,
    strip_tags,
    strip_punctuation,
    strip_multiple_whitespaces,
    strip_numeric,
    remove_stopwords,
)
from .google_nlp import GoogleNLP

logger = logging.getLogger(__name__)


class NamedEntityRecognizer():

    english_model_small = 'en_core_web_sm'
    allowed_spacy_ner_labels = [
        'PERSON',
        'ORG',
        'GPE',
        'EVENT',
        'WORK_OF_ART',
        'LAW',
        'PRODUCT',
    ]

    def __init__(self):
        self.nlp = spacy.load(self.english_model_small)

    def spacy_doc_to_entity_list(self, doc):
        """
        Performs spacy's nlp on a document.
        Collects all the entities and returns them as list.
        """
        doc = self.nlp(doc)
        entity_list = []
        for ent in doc.ents:
            if ent.label_ in self.allowed_spacy_ner_labels:
                entity_list.append(ent.text)

        # use repetition as indicator of salience for spacy
        entity_counts = {}
        for entity in entity_list:
            if entity not in entity_counts:
                entity_counts[entity] = 1
            else:
                entity_counts[entity] += 1

        return [entity for entity, __ in sorted(entity_counts.items(), key=lambda item: item[1], reverse=True)]

    def google_doc_to_entity_list(self, doc,
                                  restricted_entities_list=None,
                                  filter_wikipedia_only=False,
                                  filter_proper_only=False,
                                  salience_filter=0,
                                  fallback_if_empty=False):
        """
        Calls GoogleNLP to analyze entities and
        then parses response to list.
        """
        entities = GoogleNLP.analyze_entities(doc)
        if entities is None:
            return None

        original_entity_names = [entity.name for entity in entities]

        # filters by salience_filter
        entities = [
            entity for entity in entities if
            entity.salience >= salience_filter
        ]

        # filters by restricted list of entity types
        if restricted_entities_list is not None:
            entities = [
                entity for entity in entities if
                entity.type in restricted_entities_list
            ]

        # filters by entities that have wikipedia article
        if filter_wikipedia_only:
            entities = [
                entity for entity in entities if
                'wikipedia_url' in entity.metadata
            ]

        # filters by entities recognized as Proper entities
        if filter_proper_only:
            proper_entities = []
            for entity in entities:
                for mention in entity.mentions:
                    if mention.type_ == GoogleNLP.PROPER_MENTION:
                        proper_entities.append(entity)
                        break
            entities = proper_entities

        if fallback_if_empty:
            if len(entities) == 0:
                return original_entity_names

        return [
            entity.name for entity in entities
        ]

    def doc_to_entity_list(self, doc):
        """
        Tries GoogleNLP for analyze entities
        and falls back to Spacy.
        """

        entity_list = self.google_doc_to_entity_list(
            doc,
            restricted_entities_list=GoogleNLP.ALLOWED_NER_TYPES,
            filter_wikipedia_only=True,
            filter_proper_only=False,
            fallback_if_empty=True,
        )

        if (entity_list is not None and len(entity_list) > 0):
            return entity_list

        entity_list = self.spacy_doc_to_entity_list(doc)
        return entity_list

    def doc_to_multiple_entity_lists(self, doc, entity_type_list):
        """
        Returns multiple lists of the given types,
        in a dictionary according to type.
        """
        try:
            named_entities = GoogleNLP.analyze_entities(doc)
        except RuntimeError as e:
            named_entities = None

        logger.info(
            "INFO " +
            "GOOGLE ENTITIES: %s", named_entities
        )
        entity_results = {}
        if named_entities is None:
            named_entities = self.spacy_doc_to_entity_list(doc)
            for entity_type in entity_type_list:
                entity_results[entity_type] = named_entities
            logger.warning(
                "WARNING " +
                "SPACY ENTITIES: %s", named_entities
            )
            return entity_results

        if 'proper' in entity_type_list:
            proper_entities = []
            for entity in named_entities:
                for mention in entity.mentions:
                    if mention.type_ == GoogleNLP.PROPER_MENTION:
                        proper_entities.append(entity)
                        break
            entity_results['proper'] = [
                entity.name for entity in proper_entities
            ]

        if 'wikipedia' in entity_type_list:
            wiki_entities = []
            for entity in named_entities:
                if 'wikipedia_url' in entity.metadata:
                    wiki_url = entity.metadata['wikipedia_url']
                    wiki_entities.append((entity.name, wiki_url))

            entity_results['wikipedia'] = wiki_entities

        return entity_results

    def entities_and_root_nouns_from_string(self, doc_as_string):

        # TODO organize to remove duplication
        doc = self.nlp(doc_as_string)
        entity_list = []
        for ent in doc.ents:
            entity_list.append(ent.text)

        root_nouns = []
        for chunk in doc.noun_chunks:
            root_nouns.append(chunk.root.text)
        return entity_list, root_nouns

    def topic_to_entity_list(self, topic):
        if topic.get('doc', None) is not None:
            topic_doc = topic['doc']
        else:
            topic_doc = topic['title']
            if 'summary' in topic:
                topic_doc += " " + topic['summary']
            if topic.get('seed_links', None) is not None:
                for link in topic['seed_links']:
                    topic_doc += " " + link['title']
                    topic_doc += " " + link['snippet']
        return self.doc_to_entity_list(topic_doc)

    def topic_to_proper_wikipedia_entity_lists(self, topic):
        """
        This version uses only the title and summary to create
        the entity list, ignoring the doc that may include irrelevant
        text. 
        Then return two lists, one that is the proper names and
        one that is the the wikipedia list.
        """
        topic_doc = topic['title']
        if topic.get('summary', "") != "":
            topic_doc += "; " + topic['summary']
        return self.doc_to_multiple_entity_lists(topic_doc, ['proper', 'wikipedia'])


class SimilarityProcessor():

    english_model_small = 'en_core_web_sm'
    preprocess_filters = [
        strip_tags,
        strip_punctuation,
        strip_multiple_whitespaces,
        strip_numeric,
        remove_stopwords
    ]

    def __init__(self):
        self.nlp = spacy.load(self.english_model_small)

    # NOTE perform NER before preprocessing for similarity

    @classmethod
    def preprocess_doc(cls, to_process):
        return preprocess_string(to_process.lower(), filters=cls.preprocess_filters)

    @classmethod
    def combine_duplicate_topics(cls, normalized_list):
        """
        Takes a list of topics. Transforms them into a corpus
        (based on their 'doc') field, in order to perform
        similarity queries. Combines topics with high enough
        similarity scores, thereby removing 'duplicates'.
        """

        # prepare corpus
        documents = [cls.preprocess_doc(trend['doc'])
                     for trend in normalized_list]

        dictionary = gensim.corpora.Dictionary(documents)
        corpus = [dictionary.doc2bow(doc) for doc in documents]

        # model corpus in tfidf -> lsi
        tfidf = gensim.models.TfidfModel(corpus, normalize=True)
        corpus_tfidf = tfidf[corpus]
        lsi = gensim.models.LsiModel(
            corpus_tfidf, id2word=dictionary, num_topics=100)
        corpus_lsi = lsi[corpus_tfidf]

        # build similarity space
        lsi_index = gensim.similarities.MatrixSimilarity(corpus_lsi)

        # available_list keeps track of which topics have already been used
        # in combining into one topic. 1 for is available (has not been used)
        # and 0 for is not available (has been used)
        available_list = [1 for __ in range(0, len(normalized_list))]

        new_normalized_list = []

        for list_index, topic in enumerate(normalized_list):
            # topic has been removed as duplicate
            if available_list[list_index] == 0:
                continue

            # transform doc through the models and run similarity query
            prep_doc = cls.preprocess_doc(topic['doc'])
            topic_lsi = lsi[tfidf[dictionary.doc2bow(prep_doc)]]
            sims_query = lsi_index[topic_lsi]

            # sort sims into format
            # [...(index in normalized_list, similarity_score)...]
            sims_query = sorted(enumerate(sims_query),
                                key=lambda item: -item[1])

            sims_index = 0

            # Automatically add topic as first
            # and remove from available
            topics_to_combine = [topic]
            available_list[list_index] = 1

            # hardcoded similarity score of 0.5
            # score is in (-1, 1)
            # sims_index tracks index within the list of
            # similar topics
            # [1] is the score and [0] is the index in the
            # normalized list
            while sims_query[sims_index][1] >= 0.5:
                index_in_list = sims_query[sims_index][0]
                if available_list[index_in_list] != 0:
                    topics_to_combine.append(normalized_list[index_in_list])
                    available_list[index_in_list] = 0
                sims_index += 1

            new_normalized_list.append(cls.combine_topics(topics_to_combine))

        return new_normalized_list

    @classmethod
    def combine_topics(cls, topics):
        """
        Takes a list of normalized topics and combines them,
        creating a new topic by taking the fields of 
        the first topics to have them.
        """
        combined_topic = {}
        for topic in topics:
            if (combined_topic.get('title', None) is None and
                    topic.get('title', None) is not None):
                combined_topic['title'] = topic['title']
            if (combined_topic.get('suggested_query', None) is None and
                    topic.get('suggested_query', None) is not None):
                combined_topic['suggested_query'] = topic['suggested_query']
            if (combined_topic.get('image_url', None) is None and
                    topic.get('image_url', None) is not None):
                combined_topic['image_url'] = topic['image_url']
            if (combined_topic.get('summary', None) is None and
                    topic.get('summary', None) is not None):
                combined_topic['summary'] = topic['summary']
            if (combined_topic.get('source', None) is None and
                    topic.get('source', None) is not None):
                combined_topic['source'] = topic['source']
            if (combined_topic.get('source', None) is None and
                    topic.get('source', None) is not None):
                combined_topic['source'] = topic['source']
            if (combined_topic.get('provided_category', None) is None and
                    topic.get('provided_category', None) is not None):
                combined_topic['provided_category'] = topic['provided_category']
            if (combined_topic.get('topic_type_id', None) is None and
                    topic.get('topic_type_id', None) is not None):
                combined_topic['topic_type_id'] = topic['topic_type_id']

            # topic document should be a combination
            if (combined_topic.get('doc', None) is None and
                    topic.get('doc', None) is not None):
                combined_topic['doc'] = topic['doc']
            elif (combined_topic.get('doc', None) is not None and
                  topic.get('doc', None) is not None):
                combined_topic['doc'] += "; " + topic['doc']

            # Seed links should be added together
            if (combined_topic.get('seed_links', None) is None and
                    topic.get('seed_links', None) is not None):
                combined_topic['seed_links'] = topic['seed_links'].copy()
            elif (combined_topic.get('seed_links', None) is not None and
                  topic.get('seed_links', None) is not None):
                for link in topic['seed_links']:
                    if link['url'] not in [seed_link['url'] for seed_link in combined_topic['seed_links']]:
                        combined_topic['seed_links'].append(link)

            # Creating a txt representation of all similar topics
            if (combined_topic.get('related_topics', None) is None and
                    topic.get('title', None) is not None):
                combined_topic['related_topics'] = [
                    (topic['title'], topic.get('source', ""))]
            elif topic.get('title', None) is not None:
                combined_topic['related_topics'].append(
                    (topic['title'], topic.get('source', "")))

        num_aggregate_streams = topics[0].get('num_aggregate_streams', 1)
        # I want to normalize the way we count duplication but without
        # heavily penalizing topics with large number of streams, so I use
        # the log on the number of streams, and then also take one off the
        # topics since 1 topic should be the same regardless of the number
        # of streams.
        if num_aggregate_streams < 2:
            # Log of 1 results in divide by zero
            combined_topic['count_reduced'] = (len(topics) - 1)
        else:
            combined_topic['count_reduced'] = (
                len(topics) - 1) / math.log(num_aggregate_streams, 2)

        return combined_topic
