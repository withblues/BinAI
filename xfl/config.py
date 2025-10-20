class Config:
    class Analysis:
        THREAD_POOL_THREADS = 16

        class NLP:
            MAX_STR_LEN_BEFORE_SEQ_SPLIT = 12
            MIN_MAX_WORD_LEN = 4
            MIN_MAX_ABBR_LEN = 3
            MAX_WORD_LEN = 18

        nlp = NLP()

    analysis = Analysis()
