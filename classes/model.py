from .conllu_token import Token
import tensorflow as tf
from .algorithm import ArcEager, Sample, Transition
from .state import State
import numpy as np
import pandas as pd
import keras
import copy

def dividir_array_en_mitades(array):
    mitad = len(array) // 2
    return array[:mitad], array[mitad:]

class ParserMLP:
    """
    A Multi-Layer Perceptron (MLP) class for a dependency parser, using TensorFlow and Keras.

    This class implements a neural network model designed to predict transitions in a dependency 
    parser. It utilizes the Keras Functional API, which is more suited for multi-task learning scenarios 
    like this one. The network is trained to map parsing states to transition actions, facilitating 
    the parsing process in natural language processing tasks.

    Attributes:
        word_emb_dim (int): Dimensionality of the word embeddings. Defaults to 100.
        hidden_dim (int): Dimension of the hidden layer in the neural network. Defaults to 64.
        epochs (int): Number of training epochs. Defaults to 1.
        batch_size (int): Size of the batches used in training. Defaults to 64.

    Methods:
        train(training_samples, dev_samples): Trains the MLP model using the provided training and 
            development samples. It maps these samples to IDs that can be processed by an embedding 
            layer and then calls the Keras compile and fit functions.

        evaluate(samples): Evaluates the performance of the model on a given set of samples. The 
            method aims to assess the accuracy in predicting both the transition and dependency types, 
            with expected accuracies ranging between 75% and 85%.

        run(sents): Processes a list of sentences (tokens) using the trained model to perform dependency 
            parsing. This method implements the vertical processing of sentences to predict parser 
            transitions for each token.

        Feel free to add other parameters and functions you might need to create your model
    """

    def __init__(self,
                 word_vocab_size: int = 10000, pos_vocab_size: int = 40,
                 word_emb_dim: int = 100, pos_emb_dim: int = 100,
                 relations_size: int = 38, actions_size: int = 4,
                 n_top: int = 3, hidden_dim: int = 64,
                 
                 epochs: int = 2, batch_size: int = 64):
        """
        Initializes the ParserMLP class with the specified dimensions and training parameters.

        Parameters:
            word_emb_dim (int): The dimensionality of the word embeddings.
            hidden_dim (int): The size of the hidden layer in the MLP.
            epochs (int): The number of epochs for training the model.
            batch_size (int): The batch size used during model training.
        """
        self.arc_eager = ArcEager()

        self.word_emb_dim = word_emb_dim
        self.hidden_dim = hidden_dim
        self.epochs = epochs
        self.batch_size = batch_size

        self.word_vocab_size = word_vocab_size
        self.pos_vocab_size = pos_vocab_size
        self.output_units = actions_size
        self.relations_size = relations_size

        #Inputs layers
        inputs_words = keras.layers.Input(shape = (6,))
        inputs_pos = keras.layers.Input(shape = (6,))

        # #Embedding layers
        embedding_words = keras.layers.Embedding(input_dim = word_vocab_size, output_dim = 128, input_length=6) (inputs_words)
        embedding_pos = keras.layers.Embedding(input_dim = word_vocab_size, output_dim = 128, input_length=6) (inputs_pos)
        embedding_total = keras.layers.Concatenate()([embedding_words,embedding_pos])

        flaten = keras.layers.Flatten()(embedding_total)

        # #Dense and outputs layers
        dense = keras.layers.Dense(128, activation='relu')(flaten)
        outputs_actions = keras.layers.Dense(actions_size, activation='softmax', name="Actions")(dense)
        outputs_relations = keras.layers.Dense(44, activation='softmax', name = "Relations")(dense)
        
        self.model = keras.models.Model(inputs=[inputs_words, inputs_pos], outputs=[outputs_actions,outputs_relations])
        self.model.compile(
            optimizer = "adam",
            loss = "sparse_categorical_crossentropy",
            metrics = ["accuracy"]
        )
        
        self.model.summary()
    
    def train(self, training_samples: pd.DataFrame, dev_samples: pd.DataFrame):
        """
        Trains the MLP model using the provided training and development samples.

        This method prepares the training data by mapping samples to IDs suitable for 
        embedding layers and then proceeds to compile and fit the Keras model.

        Parameters:
            training_samples (list[Sample]): A list of training samples for the parser.
            dev_samples (list[Sample]): A list of development samples used for model validation.
        """
        training_samples[['training_samples_words', 'training_samples_pos']] = training_samples['sample_feats'].apply(lambda x: pd.Series(dividir_array_en_mitades(x)))
        dev_samples[['dev_samples_words', 'dev_samples_pos']] = dev_samples['sample_feats'].apply(lambda x: pd.Series(dividir_array_en_mitades(x)))

        history = self.model.fit(
            [np.array(training_samples["training_samples_words"].tolist()), np.array(training_samples["training_samples_pos"].tolist())],
            [np.array(training_samples["action"].to_list()), np.array(training_samples["relation"].to_list())],
            validation_data=([np.array(dev_samples["dev_samples_words"].to_list()), np.array(dev_samples["dev_samples_pos"].to_list())],[np.array(dev_samples["action"].to_list()), np.array(dev_samples["relation"].to_list())]),
            epochs=self.epochs,
            batch_size=self.batch_size,
        )
        
        return history
    
    def save(self, name):
        self.model.save(f"{name}.h5")

    def evaluate(self, samples: list['Sample']):
        """
        Evaluates the model's performance on a set of samples.

        This method is used to assess the accuracy of the model in predicting the correct
        transition and dependency types. The expected accuracy range is between 75% and 85%.

        Parameters:
            samples (list[Sample]): A list of samples to evaluate the model's performance.
        """
    
    def is_valid(self, state: State, transition: str):
        if transition == "LEFT-ARC":
            return self.arc_eager.LA_is_valid(state)
        elif transition == "RIGHT-ARC":
            return self.arc_eager.RA_is_valid(state)
        elif transition == "REDUCE":
            return self.arc_eager.REDUCE_is_valid(state)
        else:
            return True
    
    def run(self, sents: list['Token'], pre_processor):
        """
        Executes the model on a list of sentences to perform dependency parsing.

        This method implements the vertical processing of sentences, predicting parser 
        transitions for each token in the sentences.

        Parameters:
            sents (list[Token]): A list of sentences, where each sentence is represented 
                                as a list of Token objects.
        """
        # Transition dictionary to map integer values to transition types
        transition_dict = {
            0: "RIGHT-ARC",
            1: "LEFT-ARC",
            2: "SHIFT",
            3: "REDUCE"
        }

        # 1. Initialize: Create the initial state for each sentence
        batch_states = [self.arc_eager.create_initial_state(sentence) for sentence in sents]
        batch_states_dict = {i: self.arc_eager.create_initial_state(sentence) for i, sentence in enumerate(sents)}
        batch_final_arcs = [[] for _ in range(len(batch_states))]

        while batch_states:
            # 2. Feature Representation: Convert states to their corresponding list of features
            batch_state_feats, batch_state_pos = [], []
            for state in batch_states:
                sample = Sample(state, None)
                feats = sample.state_to_feats(3, 3)
                batch_state_feats.append(pre_processor.tokenizer.texts_to_sequences([feats[:6]])[0])
                batch_state_pos.append(pre_processor.tokenizer.texts_to_sequences([feats[6:]])[0])

            # 3. Model Prediction: Predict the next transition and dependency type for all current states
            batch_transitions, batch_dependencies = self.model.predict(
                [np.array(batch_state_feats), np.array(batch_state_pos)]
            )
            
            # 4. Transition Sorting: For each prediction, sort transitions by likelihood
            # and select the most likely dependency type
            batch_valid_transitions, batch_valid_dependencies = [], []
            for i in range(len(batch_transitions)):
                sorted_indices = np.argsort(batch_transitions[i])[::-1]  # Highest first
                j = 0
                most_likely_transition = transition_dict[sorted_indices[j]]
                
                # 5. Validation Check: Verify if the selected transition is valid
                while not self.is_valid(batch_states[i], most_likely_transition):
                    j += 1
                    most_likely_transition = transition_dict[sorted_indices[j]]
                    
                batch_valid_transitions.append(most_likely_transition)
                
                # Determine the corresponding dependency
                if most_likely_transition == "SHIFT" or most_likely_transition == "REDUCE":
                    batch_valid_dependencies.append(None)
                else:
                    batch_valid_dependencies.append(pre_processor.relation_dict_inversed[np.argmax(batch_dependencies[i])])

            # 6. State Update: Apply the selected transitions to update the states
            new_states = []
            for i, state in enumerate(batch_states):
                wanted_key = [key for key, value in batch_states_dict.items() if value == state]
                self.arc_eager.apply_transition(state, Transition(batch_valid_transitions[i], batch_valid_dependencies[i]))
                batch_states_dict[wanted_key[0]] = state
                new_states.append(state)

            # 7. Final State Check: Remove sentences that have reached a final state
            batch_states = []
            for state in new_states:
                if self.arc_eager.final_state(state):
                    wanted_key = [key for key, value in batch_states_dict.items() if value == state]
                    batch_final_arcs[wanted_key[0]] = state.A
                else:
                    batch_states.append(state)

        # 8. Iterative Process: Repeat steps 2 to 7 until all sentences have reached their final state.
        return batch_final_arcs
        

if __name__ == "__main__":
    
    model = ParserMLP()