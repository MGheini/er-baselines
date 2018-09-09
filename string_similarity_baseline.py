import re
import nltk
import rltk

from nltk.corpus import stopwords


class AmazonRecord(rltk.Record):
    @property
    def id(self):
        return self.raw_object['id']

    @property
    def title(self):
        return self.raw_object['title']

    @property
    def description(self):
        return self.raw_object['description']

    @property
    def manufacturer(self):
        return self.raw_object['manufacturer']

    @property
    def price(self):
        return self.raw_object['price']


class GoogleRecord(rltk.Record):
    @property
    def id(self):
        return self.raw_object['id']

    @property
    def name(self):
        return self.raw_object['name']

    @property
    def description(self):
        return self.raw_object['description']

    @property
    def manufacturer(self):
        return self.raw_object['manufacturer']

    @property
    def price(self):
        return self.raw_object['price']


if __name__ == '__main__':
    amazon_dataset = rltk.Dataset(reader=rltk.CSVReader('data/Amazon.csv'), record_class=AmazonRecord,
                                  adapter=rltk.MemoryAdapter())
    google_dataset = rltk.Dataset(reader=rltk.CSVReader('data/GoogleProducts.csv'), record_class=GoogleRecord,
                                  adapter=rltk.MemoryAdapter())

    gt = rltk.GroundTruth('idAmazon', 'idGoogleBase')
    gt.load('data/Amzon_GoogleProducts_perfectMapping.csv')
    trial = rltk.Trial(ground_truth=gt)

    candidate_pairs = rltk.get_record_pairs(amazon_dataset, google_dataset)
    ngram_tokenizer = rltk.NGramTokenizer()
    stop = set(stopwords.words('english'))
    for amazon_record, google_record in candidate_pairs:
        if not trial._ground_truth.is_member(amazon_record.id, google_record.id):
            trial._ground_truth.add_negative(amazon_record.id, google_record.id)

        amazon_price = re.findall(r'(?:\d+\.)?\d+', amazon_record.price)[0]
        google_price = re.findall(r'(?:\d+\.)?\d+', google_record.price)[0]
        if abs(float(amazon_price) - float(google_price)) < 5:  # Is this the right kind of blocking?
            if rltk.jaccard_index_similarity(ngram_tokenizer.basic(amazon_record.title, 3), ngram_tokenizer.basic(google_record.name, 3)) > 0.8:
                trial.add_result(amazon_record, google_record, True)
            else:
                amazon_whole = amazon_record.title + amazon_record.description + amazon_record.manufacturer
                google_whole = google_record.name + google_record.description + google_record.manufacturer
                amazon_whole_tokenized = nltk.word_tokenize(amazon_whole.lower())
                google_whole_tokenized = nltk.word_tokenize(google_whole.lower())
                amazon_whole_tokenized_stopped = set(amazon_whole_tokenized) - stop
                google_whole_tokenized_stopped = set(google_whole_tokenized) - stop
                if rltk.jaccard_index_similarity(amazon_whole_tokenized_stopped, google_whole_tokenized_stopped) > 0.6:
                    trial.add_result(amazon_record, google_record, True)
                    continue
            trial.add_result(amazon_record, google_record, False)
        else:
            trial.add_result(amazon_record, google_record, False)

    trial.evaluate()
    print(trial.true_positives, trial.false_positives, trial.true_negatives, trial.false_negatives,
          trial.precision, trial.recall, trial.f_measure)
