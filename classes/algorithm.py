from .state import State
from .conllu_token import Token
import copy


class Transition(object):
    """
    Class to represent a parsing transition in a dependency parser.
    
    Attributes:
    - action (str): The action to take, represented as an string constant. Actions include SHIFT, REDUCE, LEFT-ARC, or RIGHT-ARC.
    - dependency (str): The type of dependency relationship (only for LEFT-ARC and RIGHT-ARC, otherwise it'll be None), corresponding to the deprel column
    """

    def __init__(self, action: int, dependency: str = None):
        self._action = action
        self._dependency = dependency

    @property
    def action(self):
        """Return the action attribute."""
        return self._action

    @property
    def dependency(self):
        """Return the dependency attribute."""
        return self._dependency

    def __str__(self):
        return f"{self._action};{self._dependency}" if self._dependency else str(self._action)


class Sample(object):
    """
    Represents a training sample for a transition-based dependency parser. 

    This class encapsulates a parser state and the corresponding transition action 
    to be taken in that state. It is used for training models that predict parser actions 
    based on the current state of the parsing process.

    Attributes:
        state (State): An instance of the State class, representing the current parsing 
                       state at a given timestep in the parsing process.
        transition (Transition): An instance of the Transition class, representing the 
                                 parser action to be taken in the given state.

    Methods:
        state_to_feats(nbuffer_feats: int = 2, nstack_feats: int = 2): Extracts features from the parsing state.
    """

    def __init__(self, state: State, transition: Transition):
        """
        Initializes a new instance of the Sample class.

        Parameters:
            state (State): The current parsing state.
            transition (Transition): The transition action corresponding to the state.
        """
        self._state = state
        self._transition = transition

    @property
    def state(self):
        """
        Retrieves the current parsing state of the sample.

        Returns:
            State: The current parsing state in this sample.
        """
        return self._state


    @property
    def transition(self):
        """
        Retrieves the transition action of the sample.

        Returns:
            Transition: The transition action representing the parser's decision at this sample's state.
        """
        return self._transition
    

    def state_to_feats(self, nbuffer_feats: int = 2, nstack_feats: int = 2):
        """
        Extracts features from a given parsing state for use in a transition-based dependency parser.

        This function generates a feature representation from the current state of the parser, 
        which includes features from both the stack and the buffer. The number of features from 
        the stack and the buffer can be specified.

        Parameters:
            nbuffer_feats (int): The number of features to extract from the buffer.
            nstack_feats (int): The number of features to extract from the stack.

        Returns:
            list[str]: A list of extracted features. The features include the words and their 
                    corresponding UPOS (Universal Part-of-Speech) tags from the specified number 
                    of elements in the stack and buffer. The format of the feature list is as follows:
                    [Word_stack_n,...,Word_stack_0, Word_buffer_0,...,Word_buffer_m, 
                        UPOS_stack_n,...,UPOS_stack_0, UPOS_buffer_0,...,UPOS_buffer_m]
                    where 'n' is nstack_feats and 'm' is nbuffer_feats.

        Examples:
            Example 1:
                State: Stack (size=1): (0, ROOT, ROOT_UPOS)
                    Buffer (size=13): (1, Distribution, NOUN) | ... | (13, ., PUNCT)
                    Arcs (size=0): []

                Output: ['<PAD>', 'ROOT', 'Distribution', 'of', '<PAD>', 'ROOT_UPOS', 'NOUN', 'ADP']

            Example 2:
                State: Stack (size=2): (0, ROOT, ROOT_UPOS) | (1, Distribution, NOUN)
                    Buffer (size=10): (4, license, NOUN) | ... | (13, ., PUNCT)
                    Arcs (size=2): [(4, 'det', 3), (4, 'case', 2)]

                Output: ['ROOT', 'Distribution', 'license', 'does', 'ROOT_UPOS', 'NOUN', 'NOUN', 'AUX']
        """
        aux_stack = self.state.S
        aux_buffer = self.state.B
        
        if len(aux_stack) > nstack_feats:
            aux_stack = aux_stack[-nstack_feats:]
            
        if len(aux_buffer) > nbuffer_feats:
            aux_buffer = aux_buffer[:nbuffer_feats]
        
        stack_word_list = [token.form for token in aux_stack]
        stack_upos_list = [token.upos for token in aux_stack]
        buffer_word_list = [token.form for token in aux_buffer]
        buffer_upos_list = [token.upos for token in aux_buffer]
        
        stack_word_list = ['<PAD>']*(nstack_feats-len(stack_word_list)) + stack_word_list
        stack_upos_list = ['<PAD>']*(nstack_feats-len(stack_upos_list)) + stack_upos_list
        buffer_word_list = buffer_word_list + ['<PAD>']*(nbuffer_feats-len(buffer_word_list))
        buffer_upos_list = buffer_upos_list + ['<PAD>']*(nbuffer_feats-len(buffer_upos_list))
        
        return stack_word_list + buffer_word_list + stack_upos_list + buffer_upos_list
            
        
    

    def __str__(self):
        """
        Returns a string representation of the sample, including its state and transition.

        Returns:
            str: A string representing the state and transition of the sample.
        """
        return f"Sample - State:\n\n{self._state}\nSample - Transition: {self._transition}"



class ArcEager():

    """
    Implements the arc-eager transition-based parsing algorithm for dependency parsing.

    This class includes methods for creating initial parsing states, applying transitions to 
    these states, and determining the correct sequence of transitions for a given sentence.

    Methods:
        create_initial_state(sent: list[Token]): Creates the initial state for a given sentence.
        final_state(state: State): Checks if the current parsing state is a valid final configuration.
        LA_is_valid(state: State): Determines if a LEFT-ARC transition is valid for the current state.
        LA_is_correct(state: State): Determines if a LEFT-ARC transition is correct for the current state.
        RA_is_correct(state: State): Determines if a RIGHT-ARC transition is correct for the current state.
        RA_is_valid(state: State): Checks if a RIGHT-ARC transition is valid for the current state.
        REDUCE_is_correct(state: State): Determines if a REDUCE transition is correct for the current state.
        REDUCE_is_valid(state: State): Determines if a REDUCE transition is valid for the current state.
        oracle(sent: list[Token]): Computes the gold transitions for a given sentence.
        apply_transition(state: State, transition: Transition): Applies a given transition to the current state.
        gold_arcs(sent: list[Token]): Extracts gold-standard dependency arcs from a sentence.
    """

    LA = "LEFT-ARC"
    RA = "RIGHT-ARC"
    SHIFT = "SHIFT"
    REDUCE = "REDUCE"

    def create_initial_state(self, sent: list['Token']) -> State:
        """
        Creates the initial state for the arc-eager parsing algorithm given a sentence.

        This function initializes the parsing state, which is essential for beginning the parsing process. 
        The initial state consists of a stack (initially containing only the root token), a buffer 
        (containing all tokens of the sentence except the root), and an empty set of arcs.

        Parameters:
            sent (list[Token]): A list of 'Token' instances representing the sentence to be parsed. 
                                The first token in the list should typically be a 'ROOT' token.

        Returns:
            State: The initial parsing state, comprising a stack with the root token, a buffer with 
                the remaining tokens, and an empty set of arcs.
        """
        return State([sent[0]], sent[1:], set([]))
    
    def final_state(self, state: State) -> bool:
        """
        Checks if the curent parsing state is a valid final configuration, i.e., the buffer is empty

            Parameters:
                state (State): The parsing configuration to be checked

            Returns: A boolean that indicates if state is final or not
        """
        return len(state.B) == 0

    def LA_is_valid(self, state: State) -> bool:
        """
        Determines if a LEFT-ARC (LA) transition is valid for the current parsing state.

        A LEFT-ARC transition is valid if certain preconditions are met in the parser's state.
        This typically involves checking the current state of the stack and buffer in the parser.

        Parameters:
            state (State): The current state of the parser, including the stack and buffer.

        Returns:
            bool: True if a LEFT-ARC transition is valid in the current state, False otherwise.
        """
        if state.S[-1].id == 0:
            return False
        elif any((state.B[0].id, state.S[-1].id) == (a[0], a[2]) for a in state.A):
            return False
        else:
            return True
        
        

    def LA_is_correct(self, state: State) -> bool:
        """
        Determines if a LEFT-ARC (LA) transition is the correct action for the current parsing state.

        This method checks if applying a LEFT-ARC transition will correctly reflect the dependency
        structure of the sentence being parsed, based on the current state of the parser.

        Parameters:
            state (State): The current state of the parser, including the stack and buffer.

        Returns:
            bool: True if a LEFT-ARC transition is the correct action in the current state, False otherwise.
        """
        if state.S[-1].head == state.B[0].id:
            return True
        else:
            return False
    
    def RA_is_correct(self, state: State) -> bool:
        """
        Determines if a RIGHT-ARC (RA) transition is the correct action for the current parsing state.

        This method assesses whether applying a RIGHT-ARC transition aligns with the correct 
        dependency structure of the sentence, based on the parser's current state.

        Parameters:
            state (State): The current state of the parser, including the stack and buffer.

        Returns:
            bool: True if a RIGHT-ARC transition is the correct action in the current state, False otherwise.
        """
        if state.B[0].head == state.S[-1].id:
            return True
        else:
            return False

    def RA_is_valid(self, state: State) -> bool:
        """
        Checks the preconditions in order to apply a right-arc (RA) transition.

        A RIGHT-ARC transition is valid under certain conditions related to the state of the stack
        and buffer in the parser. This method evaluates these conditions to determine if a RIGHT-ARC
        can be applied.

        Parameters:
            state (State): The current parsing state of the parser.

        Returns:
            bool: True if a RIGHT-ARC transition can be validly applied in the current state, False otherwise.
        """
        if any((state.S[-1].id, state.B[0].id) == (a[0], a[2]) for a in state.A):
            return False
        else:
            return True

    def REDUCE_is_correct(self, state: State) -> bool:
        """
        Determines if applying a REDUCE transition is the correct action for the current parsing state.

        A REDUCE transition is correct if there is no word in the buffer (state.B) whose head 
        is the word on the top of the stack (state.S[-1]). This method checks this condition 
        against the current state of the parser.

        REDUCE can be only correct iff for every word in the buffer, 
        no word has as head the top word from the stack 

        Parameters:
            state (State): The current state of the parser, including the stack and buffer.

        Returns:
            bool: True if a REDUCE transition is the correct action in the current state, False otherwise.
        """
        #It is correct to do if there is no word in the state buffer  (state.B) which head is 
        #the word on the top of the stack (state.S[-1])
        for word in state.B:
            if word.head == state.S[-1].id:
                return False
        
        for arc in state.A:
            if state.S[-1].head in arc: #peligro inminente
                return True
        
        return False

    def REDUCE_is_valid(self, state: State) -> bool:
        """
        Determines if a REDUCE transition is valid for the current parsing state.

        This method checks if the preconditions for applying a REDUCE transition are met in 
        the current state of the parser. This typically involves assessing the state of the 
        stack and buffer.

        Parameters:
            state (State): The current state of the parser, including the stack and buffer.

        Returns:
            bool: True if a REDUCE transition is valid in the current state, False otherwise.
        """
        for arc in state.A:
            if state.S[-1].id in arc:
                return True
        return False

    def oracle(self, sent: list['Token']) -> list['Sample']:
        """
        Computes the gold transitions to take at each parsing step, given an input dependency tree.

        This function iterates through a given sentence, represented as a dependency tree, to generate a sequence 
        of gold-standard transitions. These transitions are what an ideal parser should predict at each step to 
        correctly parse the sentence. The function checks the validity and correctness of possible transitions 
        at each step and selects the appropriate one based on the arc-eager parsing algorithm. It is primarily 
        used for later training a dependency parser.

        Parameters:
            sent (list['Token']): A list of 'Token' instances representing a dependency tree. Each 'Token' 
                        should contain information about a word/token in a sentence.

        Returns:
            samples (list['Sample']): A list of Sample instances. Each Sample stores an state instance and a transition instance
            with the information of the outputs to predict (the transition and optionally the dependency label)
        """

        state = self.create_initial_state(sent) 

        samples = [] #Store here all training samples for sent

        #Applies the transition system until a final configuration state is reached
        while not self.final_state(state):
            # print(state)
            if self.LA_is_valid(state) and self.LA_is_correct(state):
                #Add current state 'state' (the input) and the transition taken (the desired output) to the list of samples
                #Update the state by applying the LA transition using the function apply_transition
                s = state.S[-1]
                transition = Transition(self.LA, s.dep)
                
                
                deep_copy = copy.deepcopy(state)
                samples.append(Sample(deep_copy, transition))
                self.apply_transition(state, transition)
                # print(transition)

            elif self.RA_is_valid(state) and self.RA_is_correct(state):
                #Add current state 'state' (the input) and the transition taken (the desired output) to the list of samples
                #Update the state by applying the RA transition using the function apply_transition
                b = state.B[0]
                transition = Transition(self.RA, b.dep)

                deep_copy = copy.deepcopy(state)
                samples.append(Sample(deep_copy, transition))
                self.apply_transition(state, transition)
                # print(transition)

            elif self.REDUCE_is_valid(state) and self.REDUCE_is_correct(state):
                #Add current state 'state' (the input) and the transition taken (the desired output) to the list of samples
                #Update the state by applying the REDUCE transition using the function apply_transition
                transition = Transition(self.REDUCE)
                deep_copy = copy.deepcopy(state)
                samples.append(Sample(deep_copy, transition))
                self.apply_transition(state, transition)
                # print(transition)
            else:
                #If no other transiton can be applied, it's a SHIFT transition
                transition = Transition(self.SHIFT)
                #Add current state 'state' (the input) and the transition taken (the desired output) to the list of samples
                deep_copy = copy.deepcopy(state)
                samples.append(Sample(deep_copy, transition))
                #Update the state by applying the SHIFT transition using the function apply_transition
                self.apply_transition(state,transition)
                # print(transition)


        #When the oracle ends, the generated arcs must
        #match exactly the gold arcs, otherwise the obtained sequence of transitions is not correct
        assert self.gold_arcs(sent) == state.A, f"\nGold arcs: {self.gold_arcs(sent)}\n Generated arcs :{state.A}"
    
        return samples         
    

    def apply_transition(self, state: State, transition: Transition):
        """
        Applies a given transition to the current parsing state.

        This method updates the state based on the type of transition - LEFT-ARC, RIGHT-ARC, 
        or REDUCE - and the validity of applying such a transition in the current context.

        Parameters:
            state (State): The current parsing state, which includes a stack (S), 
                        a buffer (B), and a set of arcs (A).
            transition (Transition): The transition to be applied, consisting of an action
                                    (LEFT-ARC, RIGHT-ARC, REDUCE) and an optional dependency label (only for LEFT-ARC and RIGHT-arc).

        Returns:
            None; the state is modified in place.
        """

        # Extract the action and dependency label from the transition
        t = transition.action
        dep = transition.dependency

        # The top item on the stack and the first item in the buffer
        s = state.S[-1] if state.S else None  # Top of the stack
        b = state.B[0] if state.B else None   # First in the buffer

        if t == self.LA and self.LA_is_valid(state):
            # LEFT-ARC transition logic: to be implemented
            # Add an arc to the state from the top of the buffer to the top of the stack
            # Remove from the state the top word from the stack
            state.A.add((b.id, dep, s.id))
            del state.S[-1]

        elif t == self.RA and self.RA_is_valid(state): 
            # RIGHT-ARC transition
            # Add an arc to the state from the stack top to the buffer head with the specified dependency
            # Move from the state the buffer head to the stack
            # Remove from the state the first item from the buffer
            state.A.add((s.id, dep, b.id))
            state.S.append(b)
            del state.B[0]

        elif t == self.REDUCE and self.REDUCE_is_valid(state): 
            # REDUCE transition logic: to be implemented
            # Remove from state the word from the top of the stack
            del state.S[-1]

        else:
            # SHIFT transition logic: Already implemented! Use it as a basis to implement the others
            #This involves moving the top of the buffer to the stack
            state.S.append(b) 
            del state.B[:1]
    


    def gold_arcs(self, sent: list['Token']) -> set:
        """
        Extracts and returns the gold-standard dependency arcs from a given sentence.

        This function processes a sentence represented by a list of Token objects to extract the dependency relations 
        (arcs) present in the sentence. Each Token object should contain information about its head (the id of the 
        parent token in the dependency tree), the type of dependency, and its own id. The function constructs a set 
        of tuples, each representing a dependency arc in the sentence.

        Parameters:
            sent (list[Token]): A list of Token objects representing the sentence. Each Token object contains 
                                information about a word or punctuation in a sentence, including its dependency 
                                relations and other annotations.

        Returns:
            gold_arcs (set[tuple]): A set of tuples, where each tuple is a triplet (head_id, dependency_type, dependent_id). 
                                    This represents all the gold-standard dependency arcs in the sentence. The head_id and 
                                    dependent_id are integers representing the respective tokens in the sentence, and 
                                    dependency_type is a string indicating the type of dependency relation.
        """
        gold_arcs = set([])
        for token in sent[1:]:
            gold_arcs.add((token.head, token.dep, token.id))

        return gold_arcs


   


if __name__ == "__main__":


    print("**************************************************")
    print("*               Arc-eager function               *")
    print("**************************************************\n")

    print("Creating the initial state for the sentence: 'The cat is sleeping.' \n")

    tree = [
        Token(0, 'ROOT', 'ROOT', 'ROOT_UPOS', 'ROOT_CPOS', 'ROOT_FEATS', '_', '_'),
        Token(1, 'Distribution', 'distribution', 'NOUN', 'S', 'Number=Sing', 7, 'nsubj'),
        Token(2, 'of', 'of', 'ADP', 'E', '_', 4, 'case'),
        Token(3, 'this', 'this', 'DET', 'DD', 'Number=Sing|PronType=Dem', 4, 'det'),
        Token(4, 'license', 'license', 'NOUN', 'S', 'Number=Sing', 1, 'nmod'),
        Token(5, 'does', 'do', 'AUX', 'VM', 'Mood=Ind|Number=Sing|Person=3|Tense=Pres|VerbForm=Fin', 7, 'aux'),
        Token(6, 'not', 'not', 'PART', 'PART', 'Polarity=Neg', 7, 'advmod'),
        Token(7, 'create', 'create', 'VERB', 'V', 'Mood=Ind|Number=Plur|Tense=Pres|VerbForm=Fin', 0, 'root'),
        Token(8, 'an', 'a', 'DET', 'RI', 'Definite=Ind|Number=Sing|PronType=Art', 12, 'det'),
        Token(9, 'attorney', 'attorney', 'NOUN', 'S', 'Number=Sing', 12, 'nmod'),
        Token(10, '-', '-', 'PUNCT', 'FF', '_', 9, 'punct'),
        Token(11, 'client', 'client', 'NOUN', 'S', 'Number=Sing', 9, 'compound'),
        Token(12, 'relationship', 'relationship', 'NOUN', 'S', 'Number=Sing', 7, 'obj'),
        Token(13, '.', '.', 'PUNCT', 'FS', '_', 7, 'punct')
    ]

    arc_eager = ArcEager()
    print("Initial state")
    state = arc_eager.create_initial_state(tree)
    print(state)

    #Checking that is a final state
    print (f"Is the initial state a valid final state (buffer is empty)? {arc_eager.final_state(state)}\n")

    arc_eager.oracle(tree)
    # # Applying a SHIFT transition
    # transition1 = Transition(arc_eager.SHIFT)
    # arc_eager.apply_transition(state, transition1)
    # print("State after applying the SHIFT transition:")
    # print(state, "\n")

    # #Obtaining the gold_arcs of the sentence with the function gold_arcs
    # gold_arcs = arc_eager.gold_arcs(tree)
    # print (f"Set of gold arcs: {gold_arcs}\n\n")


    # print("**************************************************")
    # print("*  Creating instances of the class Transition    *")
    # print("**************************************************")

    # # Creating a SHIFT transition
    # shift_transition = Transition(ArcEager.SHIFT)
    # # Printing the created transition
    # print(f"Created Transition: {shift_transition}")  # Output: Created Transition: SHIFT

    # # Creating a LEFT-ARC transition with a specific dependency type
    # left_arc_transition = Transition(ArcEager.LA, "nsubj")
    # # Printing the created transition
    # print(f"Created Transition: {left_arc_transition}")

    # # Creating a RIGHT-ARC transition with a specific dependency type
    # right_arc_transition = Transition(ArcEager.RA, "amod")
    # # Printing the created transition
    # print(f"Created Transition: {right_arc_transition}")

    # # Creating a REDUCE transition
    # reduce_transition = Transition(ArcEager.REDUCE)
    # # Printing the created transition
    # print(f"Created Transition: {reduce_transition}")  # Output: Created Transition: SHIFT

    # print()
    # print("**************************************************")
    # print("*     Creating instances of the class  Sample    *")
    # print("**************************************************")

    # # For demonstration, let's create a dummy State instance
    # state = arc_eager.create_initial_state(tree)  # Replace with actual state initialization as per your implementation

    # # Create a Transition instance. For example, a SHIFT transition
    # shift_transition = Transition(ArcEager.SHIFT)

    # # Now, create a Sample instance using the state and transition
    # sample_instance = Sample(state, shift_transition)
    # # To display the created Sample instance
    # print("Sample:\n", sample_instance)