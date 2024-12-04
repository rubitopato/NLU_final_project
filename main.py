from classes.conllu_reader import ConlluReader
from classes.algorithm import ArcEager
from classes.preprocessor import PreProcessor
import numpy as np
import pandas as pd
from classes.model import ParserMLP
from classes.postprocessor import PostProcessor

np.random.seed(2)

# #Create and instance of the ArcEager
arc_eager = ArcEager()

def read_file(reader, path, inference):
    trees = reader.read_conllu_file(path, inference)
    print(f"Read a total of {len(trees)} sentences from {path}")
    print (f"Printing the first sentence of the training set... trees[0] = {trees[0]}")
    for token in trees[0]:
        print (token)
    print ()
    return trees
            
def change_token_information(sentence_arcs, sentence):
    sentence_final = []
    for token in sentence:
        for arc in sentence_arcs:
            if token._id == arc[2]:
                token._head = arc[0]
                token._dep = arc[1]
        sentence_final.append(token)
    return sentence_final[1:]
    
"""
ALREADY IMPLEMENTED
Read and convert CoNLLU files into tree structures
"""
# Initialize the ConlluReader
reader = ConlluReader()
train_trees = read_file(reader,path="dataset/en_partut-ud-train_clean.conllu", inference=False)
dev_trees = read_file(reader,path="dataset/en_partut-ud-dev_clean.conllu", inference=False)
test_trees = read_file(reader,path="dataset/en_partut-ud-test_clean.conllu", inference=True)

"""
We remove the non-projective sentences from the training and development set,
as the Arc-Eager algorithm cannot parse non-projective sentences.

We don't remove them from test set set, because for those we only will do inference
"""
train_trees = reader.remove_non_projective_trees(train_trees)
dev_trees = reader.remove_non_projective_trees(dev_trees)
# test_trees = reader.remove_non_projective_trees(test_trees)
# print(train_trees)
# print(train_trees)
print ("Total training trees after removing non-projective sentences", len(train_trees))
print ("Total dev trees after removing non-projective sentences", len(dev_trees))
print ("Total test trees after removing non-projective sentences", len(test_trees))


print ("\n ------ TODO: Implement the rest of the assignment ------")

# Complete the ArcEager algorithm class.
# 1. Implement the 'oracle' function and auxiliary functions to determine the correct parser actions.
#    Note: The SHIFT action is already implemented as an example.
#    Additional Note: The 'create_initial_state()', 'final_state()', and 'gold_arcs()' functions are already implemented.

pre_processor = PreProcessor()
# 2. Use the 'oracle' function in ArcEager to generate all training samples, creating a dataset for training the neural model.
training_set = pre_processor.create_dataset(arc_eager=arc_eager, data_trees=train_trees)

# 3. Utilize the same 'oracle' function to generate development samples for model tuning and evaluation.
dev_set = pre_processor.create_dataset(arc_eager, dev_trees)

# TODO: Implement the 'state_to_feats' function in the Sample class.
# This function should convert the current parser state into a list of features for use by the neural model classifier.

# Encoding
training_set = pre_processor.encode_relations(training_set, True)
dev_set = pre_processor.encode_relations(dev_set)
# Training tokenizer
pre_processor.train_tokenizer(training_set)

# Encoding with tokenizer
encoded_training_set = pre_processor.encode_data(training_set)
encoded_dev_set = pre_processor.encode_data(dev_set)

full_training_dataframe = pd.concat(encoded_training_set, ignore_index=True)
full_dev_dataframe = pd.concat(encoded_dev_set, ignore_index=True)

# TODO: Define and implement the neural model in the 'model.py' module.
model = ParserMLP(pre_processor.word_index_length, pre_processor.relation_dict_length, epochs=2)

# 1. Train the model on the generated training dataset.
model.train(full_training_dataframe, full_dev_dataframe)
# model.save("ModeloTest")
# 2. Evaluate the model's performance using the development dataset.
# 3. Conduct inference on the test set with the trained model.
# 4. Save the parsing results of the test set in CoNLLU format for further analysis.
batch_final_arcs = model.run(test_trees, pre_processor)

for i, sentence_arcs in enumerate(batch_final_arcs):
    final_sentence = change_token_information(sentence_arcs, test_trees[i])
    with open('dataset/output.conllu', 'a') as archivo:
        for token in final_sentence:
            archivo.write(str(token) + '\n')
        archivo.write('\n')


# TODO: Utilize the 'postprocessor' module (already implemented).
post_processor = PostProcessor()

# 1. Read the output saved in the CoNLLU file and address any issues with ill-formed trees.
# 2. Specify the file path: path = "<YOUR_PATH_TO_OUTPUT_FILE>"

reader = ConlluReader()
print ("An example of a corrupted tree, before postprocessing:")
post_processing_path = "dataset/corrupted_output.conllu"
ill_trees = reader.read_conllu_file(post_processing_path)

# 3. Process the file: trees = postprocessor.postprocess(path)
print("\nSTARTING POSTPROCESSING\n")
processed_trees = post_processor.postprocess(post_processing_path)

# 4. Save the processed trees to a new output file.
with open('dataset/output_fixed.conllu', 'w') as archivo:
    for tree in processed_trees:
        for token in tree:
            if str(token)[0] != "0":
                archivo.write(str(token) + "\n")
        archivo.write('\n')
