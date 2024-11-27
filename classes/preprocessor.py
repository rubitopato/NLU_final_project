import pandas as pd
import keras.preprocessing

class PreProcessor:

    def __init__(self):
        self.tokenizer = keras.preprocessing.text.Tokenizer(oov_token='<OOV>')

    def train_tokenizer(self, train_input_data):

        encoded_datasets = []
        all_tokens = set()

        for df in train_input_data:
            for data in df.iterrows():
                for token in data[1].sample_feats:
                    all_tokens.add(token)

        self.tokenizer.fit_on_texts(all_tokens)
        self.word_index_length = len(self.tokenizer.word_index)+1   
        print(self.word_index_length)

    def encode_actions(self, action):
        action_dict = {
            "RIGHT-ARC": 1,
            "LEFT-ARC": 2,
            "SHIFT": 3,
            "REDUCE": 4
        }
        return action_dict.get(action, None) 

    def encode_relations(self, dataset_array):
        relation_dict = self.get_relation_dict(dataset_array)

        encoded_datasets = []
        for df in dataset_array:
            if 'relation' in df.columns:
                df = df.copy()
                df['relation'] = df['relation'].map(relation_dict)
            encoded_datasets.append(df)
        
        return encoded_datasets

    def get_relation_dict(self, dataset_array):
        unique_relations = set()
        
        for df in dataset_array:
            if 'relation' in df.columns:
                unique_relations.update(df['relation'].unique())
        
        relation_dict = {relation: index for index, relation in enumerate(unique_relations, start=1)}
        
        self.relation_dict_length = len(relation_dict)
        return relation_dict

    def get_training_sentences(self, dataset_array):
        phrases = []

        for df in dataset_array:
            if 'buffer' in df.columns and not df.empty:
                first_row_buffer = df.iloc[0]['buffer']
                if isinstance(first_row_buffer, list):
                    phrase = " ".join(first_row_buffer)
                    phrases.append(phrase)
        
        concatenated_df = pd.DataFrame({'sentences': phrases})

        return concatenated_df
    
    def encode_data(self, dataset_array):

        encoded_datasets = []

        for df in dataset_array:
                encoded_df = df.copy()

                encoded_df['sample_feats'] = encoded_df['sample_feats'].apply(
                    lambda words: self.tokenizer.texts_to_sequences([words])[0] if isinstance(words, list) else words
                )
                
                encoded_datasets.append(encoded_df)
        
        return encoded_datasets


    def create_dataset(self, arc_eager, data_trees):
        dataset = []
        for tree in data_trees:
            sentence_result = arc_eager.oracle(tree)
            
            actions = []
            sample_feats_array = []
            relations = []
            
            for data_step in sentence_result:
                
                sample_feats = data_step.state_to_feats(nbuffer_feats = 3,
                                         nstack_feats = 3)
                sample_feats_array.append(sample_feats)


                # Actions/Relations
                splitted_transition = data_step.transition.__str__().split(";")
                action = self.encode_actions(splitted_transition[0]) 
                relation = splitted_transition[1] if len(splitted_transition) > 1 else None
                
                actions.append(action)
                relations.append(relation)
            
            sentence_df = pd.DataFrame({
                'sample_feats':sample_feats_array,
                'action': actions,
                'relation': relations,
            })
            dataset.append(sentence_df)
        
        return dataset
