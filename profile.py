class profile:
    """ class profile """
    dataset = []
    setting = []
    method = []
    result = []

    def __init__(self, dataset, setting, method, result):
        self.dataset = dataset
        self.setting = setting
        self.method = method
        self.result = result

    def run_baseline(self):
        # load dataset
        self.setting.load()
        data = self.dataset.load(self.setting.fold, self.setting.sample_rate)
        embedding_feature = self.method.run(data, self.setting.fold, self.setting.sample_rate)
        self.result.run(embedding_feature, self.setting.fold, self.setting.sample_rate)
