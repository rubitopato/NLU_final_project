import pandas as pd
import keras.preprocessing

class PreProcessor:

    def __init__(self):
        self.tokenizer = keras.preprocessing.text.Tokenizer(oov_token='<OOV>')

    def train_tokenizer(self, train_input_data):

        data = [s for s in train_input_data['sentences'].to_list()]

        self.tokenizer.fit_on_texts(data)
        self.word_index = len(self.tokenizer.word_index)+1   
        print(self.tokenizer.word_index)

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
            # Verificar si las columnas 'stack' y 'buffer' existen
            if 'stack' in df.columns and 'buffer' in df.columns:
                # Hacer una copia del DataFrame para no modificar el original
                encoded_df = df.copy()

                # Codificar la columna 'stack' usando texts_to_sequences
                encoded_df['stack'] = encoded_df['stack'].apply(
                    lambda words: self.tokenizer.texts_to_sequences([words])[0] if isinstance(words, list) else words
                )
                
                # Codificar la columna 'buffer' usando texts_to_sequences
                encoded_df['buffer'] = encoded_df['buffer'].apply(
                    lambda words: self.tokenizer.texts_to_sequences([words])[0] if isinstance(words, list) else words
                )

                # Añadir el DataFrame codificado a la lista de resultados
                encoded_datasets.append(encoded_df)
        
        return encoded_datasets


    def create_dataset(self, arc_eager, data_trees):
        dataset = []
        for tree in data_trees:
            sentence_result = arc_eager.oracle(tree)
            
            stacks = []
            buffers = []
            actions = []
            relations = []
            
            for data_step in sentence_result:
                # Buffer
                full_buffer = data_step.state.B
                buffer = [buffer_elements.__str__().split("\t")[1] for buffer_elements in full_buffer]
                
                # Stack
                full_stack = data_step.state.S
                stack = [stack_elements.__str__().split("\t")[1] for stack_elements in full_stack]
                
                # Actions/Relations
                splitted_transition = data_step.transition.__str__().split(";")
                action = self.encode_actions(splitted_transition[0]) 
                relation = splitted_transition[1] if len(splitted_transition) > 1 else None
                
                stacks.append(stack)
                buffers.append(buffer)
                actions.append(action)
                relations.append(relation)
            
            sentence_df = pd.DataFrame({
                'stack': stacks,
                'buffer': buffers,
                'action': actions,
                'relation': relations
            })
            dataset.append(sentence_df)
        
        return dataset
