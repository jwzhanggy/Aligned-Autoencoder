from setting import *
import math
from sklearn import cross_validation
import copy
import random
import pickle

# this is the exp setting class, if will do something real!
class expSettingPUCoIteration(setting):
    source_target = {'source': 'foursquare', 'target': 'twitter'}
    anchor_link_sample_rate = 1.0
    target_social_link_sample_rate = 1.0
    source_social_link_sample_rate = 1.0
    fold = 5
    recommendation_target = 'network'
    recommendation_type = 'friend'
    prefix_path = ""
    first = True
    
    
    def load_classify_evaluate_save(self):
        print "start loading data"
        
        load_data = self.settingDataset.load()
        network = load_data['network']
        social_pair = load_data['social_pair']
        
        target_network = network['foursquare']
        source_network = network['twitter']
        target_social_pair = social_pair['foursquare']
        source_social_pair = social_pair['twitter']
        
        sampled_target_network = self.sample_network(target_network, self.target_social_link_sample_rate)
        sampled_source_network = self.sample_network(source_network, self.source_social_link_sample_rate)
        sampled_target_anchor_links = self.sample_anchor_link(sampled_target_network['text'].keys(), sampled_source_network['text'].keys(), target_network['mapping'])
        sampled_source_anchor_links = self.reverse_anchor_link(sampled_target_anchor_links)
        sampled_target_network['mapping'] = sampled_target_anchor_links
        sampled_source_network['mapping'] = sampled_source_anchor_links
        
        #print anchor_pair
        print '*******************'
        if self.first:
            print "partition social links"
            self.partition_pairs(target_social_pair, "target")
            self.partition_pairs(source_social_pair, 'source')
        
        for fold_count in range(1, self.fold + 1):
            print "fold counter: " + str(fold_count)
            file_name = ""
            
            target_social_pairs = self.load_pairs("target", self.target_social_link_sample_rate, fold_count)
            source_social_pairs = self.load_pairs("source", self.source_social_link_sample_rate, fold_count)
        
            sampled_target_network['user'] = self.get_user_dict(target_social_pairs)
            sampled_source_network['user'] = self.get_user_dict(source_social_pairs)
            
            print 'start classifying'
            self.settingMethod.refresh()
            self.settingMethod.fold_count = fold_count
            self.settingMethod.target_network = sampled_target_network
            self.settingMethod.source_network = sampled_source_network
            self.settingMethod.target_social_pair = target_social_pairs
            self.settingMethod.source_social_pair = source_social_pairs
            self.settingMethod.result_obj = self.settingResult
            self.settingMethod.result_obj.fold_count = fold_count
            result = self.settingMethod.classify()
            
            print "start saving results"
            print result
            self.settingResult.fold_count = fold_count
            self.settingResult.save(result)
    
    def get_user_dict(self, pair_dicts):
        user_dict = {}
        for train_test in ['train', 'test']:
            if train_test == 'test':
                positive_negative_list = ['positive', 'negative']
            else:
                positive_negative_list = ['P', 'S', 'P_minus_S', 'U', 'U_plus_S']
            for p_n in positive_negative_list:
                pair_list = pair_dicts[train_test][p_n]
                for pair in pair_list:
                    if pair[0] not in user_dict:
                        user_dict[pair[0]] = {}
                    if pair[1] not in user_dict:
                        user_dict[pair[1]] = {}
        return user_dict
    
    def sample_network(self, network, rate):
        new_network = {'text': {}, 'time': {}, 'location': {}}
        for user in network['text']:
            word_dict = {}
            for word in network['text'][user]:
                word_dict[word] = float(network['text'][user][word]) * rate
            new_network['text'][user] = word_dict
        
        for user in network['time']:
            time_dict = {}
            for time in network['time'][user]:
                time_dict[time] = float(network['time'][user][time]) * rate
            new_network['time'][user] = time_dict
        
        for user in network['location']:
            location_list = []
            sample_index = random.sample(range(len(network['location'][user])), int(float(len(network['location'][user])) * rate))
            for index in sample_index:
                location_list.append(network['location'][user][index])
            new_network['location'][user] = location_list
        return new_network
    
    def sample_anchor_link(self, user_list1, user_list2, anchor_link_dict):
        anchor_dict = {}
        for user1 in anchor_link_dict:
            user2 = anchor_link_dict[user1]
            if user1 in user_list1 and user2 in user_list2:
                anchor_dict[user1] = user2
        sample_index = random.sample(range(len(anchor_dict)), int(float(len(anchor_dict)) * self.anchor_link_sample_rate))
        sampled_anchor_dict = {}
        for index in sample_index:
            sampled_anchor_dict[anchor_dict.keys()[index]] = anchor_dict[anchor_dict.keys()[index]]
        return sampled_anchor_dict
    
    def reverse_anchor_link(self, dict):
        reversed_dict = {}
        for user1 in dict:
            user2 = dict[user1]
            reversed_dict[user2] = user1
        return reversed_dict
        
    def load_pairs(self, link_type, rate, fold):
        print "loading pairs"
        print "recommendation type: " + str(link_type) + " link sample rate: " + str(rate) + " fold count: " + str(fold)
        input_path = self.prefix_path + link_type + "_" + str(rate) + "_" + str(fold) + "_" + str(self.fold)
        input_file = open(input_path, 'rb')
        pair = pickle.load(input_file)
        input_file.close()
        return pair
    
    def partition_pairs(self, pair, type):
        sample_rate_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        #sample_rate_list = [0.8]
        positive_pair = pair['positive']
        negative_pair = pair['negative']
        
        print "start partition pairs"
        for rate in sample_rate_list:
            for fold_count in range(1, 1 + self.fold):      
                positive_pair_copy = copy.deepcopy(positive_pair)
                negative_pair_copy = copy.deepcopy(negative_pair)
                
                #--------------------------- train positive cross avlidation ---------------------
                train_positive_index = random.sample(range(len(positive_pair_copy)), int(len(positive_pair_copy) * 0.8))
                test_positive_index = []
                for num in range(len(positive_pair_copy)):
                    if num not in train_positive_index:
                        test_positive_index.append(num)
                
                test_negative_index = []
                train_negative_index = random.sample(range(len(negative_pair_copy)), int(len(negative_pair_copy) * 0.8))
                for num in range(len(negative_pair_copy)):
                    if num not in train_negative_index:
                        test_negative_index.append(num)
                
                
                #--------------------------- test lists ---------------------------
                test_positive_list = []
                test_negative_list = []
                
                for index in test_positive_index:
                    test_positive_list.append(positive_pair[index])
                for index in test_negative_index:
                    test_negative_list.append(negative_pair[index])
                
                #----------------------------- train list: P. U. P-S, S, U+S --------
                sampled_train_positive_index = random.sample(range(len(train_positive_index)), int(len(train_positive_index) * rate))
                P_index = []
                for index in sampled_train_positive_index:
                    num = train_positive_index[index]
                    P_index.append(num)
                
                train_S_index = random.sample(range(len(P_index)), int(0.15 * len(P_index)))
                S_index = []
                P_minus_S_index = []
                for index in range(len(P_index)):
                    if index in train_S_index:
                        S_index.append(P_index[index])
                    else:
                        P_minus_S_index.append(P_index[index])
                
                P_list = []
                P_minus_S_list = []
                S_list = []
                U_list = []
                U_plus_S_list = []
                
                for index in P_index:
                    P_list.append(positive_pair[index])
                
                for index in S_index:
                    S_list.append(positive_pair[index])
                
                for index in P_minus_S_index:
                    P_minus_S_list.append(positive_pair[index])
                
                for index in train_negative_index:
                    U_list.append(negative_pair[index])
                    U_plus_S_list.append(negative_pair[index])
                
                for index in S_index:
                    U_plus_S_list.append(positive_pair[index])
                
                pair = {"train": {"P": P_list, "S": S_list, "P_minus_S": P_minus_S_list, "U": U_list, "U_plus_S": U_plus_S_list}, "test": {"positive": test_positive_list, "negative": test_negative_list}}
                #print pair
                
                output_path = self.prefix_path + type + "_" + str(rate) + "_" + str(fold_count) + "_" + str(self.fold)
                output = open(output_path, 'wb')
                pickle.dump(pair, output)
                output.close()
                