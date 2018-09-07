import rltk


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
    google_dataset = rltk.Dataset(reader=rltk.CSVReader('data/Google.csv'), record_class=GoogleRecord,
                                  adapter=rltk.MemoryAdapter())

    candidate_pairs = rltk.get_record_pairs(amazon_dataset, google_dataset)
    ngram_tokenizer = rltk.NGramTokenizer()
    for amazon_record, google_record in candidate_pairs:
        if rltk.jaccard_index_similarity(ngram_tokenizer.basic(amazon_record.title, 3), ngram_tokenizer.basic(google_record.name, 3)) > 0.8:
            print('{} and {}'.format(amazon_record.title, google_record.name))
